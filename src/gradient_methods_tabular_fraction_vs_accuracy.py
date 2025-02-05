import os.path as osp
import numpy as np
import argparse
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
import os
import h5py
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch

from src.utils.misc import get_non_zero_cols, set_random_seeds, get_path
from captum.attr import IntegratedGradients, NoiseTunnel
from src.explanation_methods.gradient_methods.local_classifier import compute_gradmethod_accuracy_per_fraction, compute_saliency_maps
from src.model.factory import ModelHandlerFactory
from src.explanation_methods.factory import ExplanationMethodHandlerFactory

from sklearn.metrics.pairwise import cosine_similarity

def cosine_distance(x, y):
    cosine_sim = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0, 0]
    return 1 - cosine_sim

def validate_distance_measure(distance_measure):
    valid_distance_measures = BallTree.valid_metrics + ["cosine"]
    assert distance_measure in valid_distance_measures, f"Invalid distance measure: {distance_measure}. Valid options are: {valid_distance_measures}"


def main(args):
    print(f"Running analysis, with following arguments: {args}")
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

    explanation_handler = ExplanationMethodHandlerFactory.get_handler(args.gradient_method, forward_func=model, multiply_by_inputs=False)

    dataset = model_handler.load_data()
    predict_fn = model_handler.predict_fn
    print("dataset of length: ", len(dataset))

    validate_distance_measure(args.distance_measure)
    distance_measure = "pyfunc" if args.distance_measure == "cosine" else args.distance_measure
    
    #   tst_feat, analysis_feat, tst_data, analysis_data, data_loader_tst
    tst_feat_for_dist, df_feat_for_dist, tst_feat_for_expl, df_feat_for_expl = explanation_handler.process_data(dataset, model_handler, args)

    print("tst_feat shape: ", tst_feat_for_dist.shape)
    print("analysis_feat shape: ", df_feat_for_dist.shape)
    print("tst_data len: ", len(tst_feat_for_expl))
    print("analysis_data len: ", len(df_feat_for_expl))

    tree = BallTree(df_feat_for_dist, metric=distance_measure) if args.distance_measure != "cosine" else BallTree(df_feat_for_dist, metric=distance_measure, func=cosine_distance)
    n_points_in_ball = np.linspace(20, int(args.max_frac* len(df_feat_for_dist)), args.num_frac, dtype=int)
    fractions = n_points_in_ball / len(df_feat_for_dist)
    
    explanation_handler = ExplanationMethodHandlerFactory.get_handler(args.gradient_method, forward_func=model, multiply_by_inputs=False)
    explainer = explanation_handler.explainer
    explanations = explanation_handler.compute_explanations(results_path=results_path, 
                                                            explainer=explainer, 
                                                            predict_fn=predict_fn, 
                                                            tst_data=tst_feat_for_expl, 
                                                            device=device, 
                                                            args=args)

    num_fractions = len(fractions)
    results = {
        "accuracy": np.zeros((num_fractions, args.max_test_points)),
        "radius": np.zeros((num_fractions, args.max_test_points)),
        "fraction_points_in_ball": fractions,
        "ratio_all_ones": np.zeros((num_fractions, args.max_test_points)),
    }

    experiment_setting = explanation_handler.get_experiment_setting(fractions, args)

    # for i in range(0, len(tst_feat), args.chunk_size):
    for i, (imgs,_,_) in enumerate(tst_feat_for_expl):
        chunk_start = i*args.chunk_size
        chunk_end = min(chunk_start + args.chunk_size, len(tst_feat_for_dist))
        print(f"Computing accuracy for chunk {i} from {chunk_start} to {chunk_end}")
        explanations_chunk = explanations[chunk_start:chunk_end]
        imgs = imgs.to(device)
        for n_closest in n_points_in_ball:
            print(f"Computing accuracy for {n_closest} closest points")
            top_labels = torch.argmax(predict_fn(imgs), dim=1).tolist()
            n_closest, acc, rad = compute_gradmethod_accuracy_per_fraction(tst_feat_for_dist[chunk_start:chunk_end], 
                                                                                top_labels,
                                                                                df_feat_for_expl, 
                                                                                explanations_chunk, 
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
        np.savez(osp.join(results_path, experiment_setting), **results)

    print("Finished computing LIME accuracy and fraction of points in the ball")

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locality Analyzer")
    parser.add_argument("--gradient_method", type=str, default = "IG", help="Which Gradient Method to use: [IG, IG+SmoothGrad]")
    parser.add_argument("--results_folder", type=str, default = "/home/grotehans/xai_locality/results/gradient_methods/integrated_gradient", help="Path to the results folder")
    parser.add_argument("--model_type", type=str, default="binary_inception_v3", help="binary_inception_v3 or inception_v3")
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--max_frac", type=float, default=0.1, help="Until when to compute the fraction of points in the ball")
    parser.add_argument("--num_frac", type=int, default=100, help="Number of fractions to compute")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--predict_threshold", type=float, default = -20, help="Threshold for classifying sample as top prediction")
    parser.add_argument("--max_test_points", type=int, default = 200)
    parser.add_argument("--chunk_size", type=int, default = 2)
                                    
    args = parser.parse_args()

    main(args)