import os.path as osp
import numpy as np
import argparse
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
from utils.misc import get_non_zero_cols, set_random_seeds, get_path
import os
from captum.attr import IntegratedGradients, NoiseTunnel
from src.explanation_methods.gradient_methods.local_classifier import compute_gradmethod_accuracy_per_fraction
from model.factory import ModelHandlerFactory
import h5py
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch

def compute_saliency_maps(explainer, predict_fn, data_loader_tst, device):
        saliency_map = []
        for i, (imgs, _, _) in enumerate(data_loader_tst):
            imgs = imgs.to(device)
            top_labels = torch.argmax(predict_fn(imgs), dim=1).tolist()
            saliency = explainer.attribute(imgs, target=top_labels).float()
            saliency_map.append(saliency)
            print("computed the first stack of saliency maps")
        return torch.cat(saliency_map, dim=0)

def main(args):
    set_random_seeds(args.random_seed)
    results_path = args.results_folder
    if not osp.exists(results_path):
        os.makedirs(results_path)
    print("saving results to: ", results_path)
    device = torch.device("cpu")
    print("device: ", device)
    model_handler = ModelHandlerFactory.get_handler(args.model_type, model_path=None)
    model = model_handler.model
    model.eval()
    model.to(device)
    dataset = model_handler.load_data()
    
    predict_fn = model_handler.predict_fn
    print("dataset of length: ", len(dataset))

    valid_distance_measures = BallTree.valid_metrics + ["cosine"]
    assert args.distance_measure in valid_distance_measures, f"Invalid distance measure: {args.distance_measure}. Valid options are: {valid_distance_measures}"
    distance_measure = args.distance_measure

    if args.distance_measure == "cosine":
        from sklearn.metrics.pairwise import cosine_similarity
        def cosine_distance(x, y):
            cosine_sim = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0, 0]
            return 1 - cosine_sim
        distance_measure = "pyfunc"
    
    # load feature vectors 
    df_feat = model_handler.load_feature_vectors()
  
    indices = np.random.permutation(len(dataset))
    tst_indices, analysis_indices = np.split(indices, [args.max_test_points])
    tst_data = Subset(dataset, tst_indices)
    analysis_data = Subset(dataset, analysis_indices)
    tst_feat, analysis_feat = np.split(df_feat[indices], [args.max_test_points])
    data_loader_tst = DataLoader(tst_data, batch_size=args.chunk_size, shuffle=False)


    # dataloader for tst_data
    print("tst_feat shape: ", tst_feat.shape)
    print("analysis_feat shape: ", analysis_feat.shape)
    print("tst_data len: ", len(tst_data))
    print("analysis_data len: ", len(analysis_data))

    tree = BallTree(analysis_feat, metric=distance_measure) if args.distance_measure != "cosine" else BallTree(df_feat, metric=distance_measure, func=cosine_distance)
    n_points_in_ball = np.linspace(20, int(args.max_frac* len(df_feat)), args.num_frac, dtype=int)
    fractions = n_points_in_ball / len(df_feat)

    explainer = IntegratedGradients(model, multiply_by_inputs=False)
    if args.gradient_method == "IG+SmoothGrad":
        explainer = NoiseTunnel(explainer)

    saliency_map_folder = osp.join(results_path, 
                                      "saliency_maps", 
                                      )
    saliency_map_file_path = osp.join(saliency_map_folder, f"saliency_map_{args.gradient_method}.h5")
    if osp.exists(saliency_map_file_path):
        with h5py.File(saliency_map_file_path, "r") as f:
            saliency_maps = f["saliency_map"][:]
        saliency_maps = torch.tensor(saliency_maps).float().to(device)
    else:
        if not osp.exists(saliency_map_folder):
            os.makedirs(saliency_map_folder)
        saliency_maps = compute_saliency_maps(explainer, predict_fn, data_loader_tst, device)
        with h5py.File(saliency_map_file_path, "w") as f:
            f.create_dataset("saliency_map", data=saliency_maps.cpu().numpy())

    num_fractions = len(fractions)
    results = {
        "accuracy": np.zeros((num_fractions, args.max_test_points)),
        "radius": np.zeros((num_fractions, args.max_test_points)),
        "fraction_points_in_ball": fractions,
        "ratio_all_ones": np.zeros((num_fractions, args.max_test_points)),
    }
    experiment_setting = f"fractions-{0}-{np.round(fractions[-1])}_grad_method-{args.gradient_method}_model_type-{args.model_type}_accuracy_fraction.npy"
    results_file_path = osp.join(results_path, experiment_setting)

    # for i in range(0, len(tst_feat), args.chunk_size):
    for i, (imgs,_,_) in enumerate(data_loader_tst):
        print(f"Computing accuracy for chunk {i} to {i+args.chunk_size}")
        chunk_start = i*args.chunk_size
        chunk_end = min(chunk_start + args.chunk_size, len(tst_feat))
        saliency_maps_chunk = saliency_maps[i:chunk_end]
        imgs = imgs.to(device)
        for n_closest in n_points_in_ball:
            print(f"Computing accuracy for {n_closest} closest points")
            top_labels = torch.argmax(predict_fn(imgs), dim=1).tolist()
            n_closest, acc, rad = compute_gradmethod_accuracy_per_fraction(tst_feat[i:i+args.chunk_size], 
                                                                                top_labels,
                                                                                analysis_data, 
                                                                                saliency_maps_chunk, 
                                                                                predict_fn, 
                                                                                n_closest, 
                                                                                tree, 
                                                                                device,
                                                                                pred_threshold=args.predict_threshold)
            fraction_idx = np.where(n_points_in_ball == n_closest)[0][0]
            results["accuracy"][fraction_idx, i:i+args.chunk_size] = acc.cpu().numpy()
            results["radius"][fraction_idx, i:i+args.chunk_size] = rad
            print(f"Finished computing accuracy for {n_closest} closest points")
        print(f"Finished computing accuracy for chunk {i} to {i+args.chunk_size}")
        np.save(results_file_path, results)

    print("Finished computing LIME accuracy and fraction of points in the ball")

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locality Analyzer")
    parser.add_argument("--gradient_method", type=str, default = "IG", help="Which Gradient Method to use: [IG, IG+SmoothGrad]")
    parser.add_argument("--results_folder", type=str, default = "/home/grotehans/xai_locality/results/gradient_methods/integrated_gradient", help="Path to the results folder")
    parser.add_argument("--model_type", type=str, default="binary_inception_v3", help="binary_inception_v3 or inception_v3")
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--max_frac", type=float, default=0.3, help="Until when to compute the fraction of points in the ball")
    parser.add_argument("--num_frac", type=int, default=100, help="Number of fractions to compute")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--predict_threshold", type=float, default = -20, help="Threshold for classifying sample as top prediction")
    parser.add_argument("--max_test_points", type=int, default = 200)
    parser.add_argument("--chunk_size", type=int, default = 2)
                                    
    args = parser.parse_args()

    main(args)