from torch_frame.gbdt import XGBoost
from torch_frame.typing import TaskType
import os.path as osp
import torch
import numpy as np
import argparse
import lime.lime_tabular
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
import random
import xgboost
from functools import partial
from utils.plotting_utils import plot_accuracy_vs_threshold, plot_accuracy_vs_fraction, plot_3d_scatter
from utils.lime_local_classifier import compute_lime_accuracy, compute_explanations
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_random_seeds(args.random_seed)

    distance_measure = args.distance_measure
    include_trn = args.include_trn
    include_val = args.include_val
    model_path = args.model_path
    data_path = args.data_path
    
    if args.model_type == "xgboost":
        from model.pytorch_frame_xgboost import load_model, load_data, predict_fn
    elif args.model_type == "lightgbm":
        from model.lightgbm import load_model, load_data, predict_fn
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Load the model using the appropriate function
    model = load_model(model_path)
    tst_feat, tst_y, val_feat, val_y, trn_feat, trn_y = load_data(model, data_path)
    feature_names = np.arange(trn_feat.shape[1])
    predict_fn_wrapped = partial(predict_fn, model=model)

    print("train, test, val feature shapes: ", trn_feat.shape, tst_feat.shape, val_feat.shape)
    df_feat = tst_feat
    df_y = tst_y
    if args.debug:
        tst_feat = tst_feat[:5]
        args.precomputed_explanations = True
        print("Debug mode: Using only the first 10 samples")

    if include_trn:
        df_feat = np.concatenate([trn_feat, df_feat], axis=0)
        df_y = np.concatenate([trn_y, df_y], axis=0)
    if include_val:
        df_feat = np.concatenate([df_feat, val_feat], axis=0)
        df_y = np.concatenate([df_y, val_y], axis=0)

    distances_pw = pairwise_distances(tst_feat, df_feat, metric=distance_measure)
    max = np.max(distances_pw)
    first_non_zero = np.min(distances_pw[np.round(distances_pw, 2) > 0])

    tree = BallTree(df_feat)  
    num_tresh = args.num_tresh if not args.debug else 2
    thresholds = np.concatenate((np.array([1e-5]), np.linspace(first_non_zero, max, num_tresh)))
    results_path = args.results_path
    df_setting = "complete_df" if include_trn and include_val else "only_test"
        
    explainer = lime.lime_tabular.LimeTabularExplainer(trn_feat, 
                                                        feature_names=feature_names, 
                                                        class_names=[0,1], 
                                                        discretize_continuous=True,
                                                        random_state=args.random_seed)
    if args.kernel_width is None:
        args.kernel_width = np.round(np.sqrt(trn_feat.shape[1]) * .75, 2) #Default value

    setting = f"normalized_data_thresholds-0-{np.round(first_non_zero)}-max{np.round(max)}num_tresh-{num_tresh}_{df_setting}_kernel_width-{args.kernel_width}_model_regr-{args.model_regressor}_accuracy_fraction.npy"
    if args.debug:
        setting = f"debug_{setting}"
        print("Start computing LIME accuracy and fraction of points in the ball")
    print("saving results to: ", setting)

    # Initialize results dictionary with numpy arrays instead of lists for better performance
    num_samples = len(tst_feat)
    num_thresholds = len(thresholds)
    results = {
        "accuracy": np.zeros((num_thresholds, num_samples)),
        "fraction_points_in_ball": np.zeros((num_thresholds, num_samples)),
        "radius": np.zeros((num_thresholds, num_samples)),
        "samples_in_ball": np.zeros((num_thresholds, num_samples)),
        "ratio_all_ones": np.zeros((num_thresholds, num_samples)),
        "thresholds": thresholds
    }
    
    if args.precomputed_explanations:
        print("Using precomputed explanations")
        explanations = np.load(osp.join(args.results_path, f"explanations/normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}.npy"), allow_pickle=True)
        print(len(explanations), "explanations loaded")
    else:
        print("Computing explanations for the test set")
        explanations = compute_explanations(explainer, tst_feat, predict_fn_wrapped)
        np.save(osp.join(args.results_path, f"explanations/normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}.npy"), explanations)
        print("Finished computing explanations for the test set")

    chunk_size = args.chunk_size
    # Split the test set into chunks
    for i in range(0, len(tst_feat), chunk_size):
        chunk_end = min(i + chunk_size, len(tst_feat))
        tst_chunk = tst_feat[i:chunk_end]
        explanations_chunk = explanations[i:chunk_end]
        chunk_results = Parallel(n_jobs=-1)(
                delayed(compute_lime_accuracy)(
                    tst_chunk, df_feat, explanations_chunk, explainer, predict_fn_wrapped, dist_threshold, tree
                )
                for dist_threshold in thresholds
            )
        chunk_results_sorted = sorted(chunk_results, key=lambda x: np.where(thresholds == x[0])[0][0])
            
        # Unpack results directly into the correct positions in the arrays
        for t, (threshold, acc, frac, rad, samp, ratio) in enumerate(chunk_results_sorted):
            threshold_idx = np.where(thresholds == threshold)[0][0]
            results["accuracy"][threshold_idx, i:chunk_end] = acc
            results["fraction_points_in_ball"][threshold_idx, i:chunk_end] = frac
            results["radius"][threshold_idx, i:chunk_end] = rad
            results["samples_in_ball"][threshold_idx, i:chunk_end] = samp
            results["ratio_all_ones"][threshold_idx, i:chunk_end] = ratio
        
        # create graphs for the accuracy and fraction of points in the ball
        plot_accuracy_vs_threshold(results["accuracy"], results["thresholds"], results_path, setting + "_accuracy_vs_threshold.pdf ")
        plot_accuracy_vs_fraction(results["accuracy"], results["fraction_points_in_ball"], results_path, setting + "_accuracy_vs_fraction.pdf")
        plot_3d_scatter(results["fraction_points_in_ball"], results["thresholds"], results["accuracy"], angles=(10, 30), cmap="tab20b", save_path=results_path, save_path=setting + "_3d_scatter.pdf")# Save intermediate results
        
        np.savez(osp.join(results_path, setting), **results)
        print(f"Processed chunk {i//chunk_size + 1}/{(len(tst_feat) + chunk_size - 1)//chunk_size}")

    print("Finished computing LIME accuracy and fraction of points in the ball")

   
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Locality Analyzer")
    parser.add_argument("--data_path", type=str,  help="Path to the data", default="/home/grotehans/xai_locality/data/")
    parser.add_argument("--model_path", type=str, help="Path to the model", default="/home/grotehans/pytorch-frame/benchmark/results/xgboost_normalized_binary_medium_6.pt")
    parser.add_argument("--model_type", type=str, default="xgboost", help="Model type, so far only 'xgboost' and 'lightgbm' is supported")
    parser.add_argument("--results_path", type=str,  help="Path to save results", default="/home/grotehans/xai_locality/results/XGBoost/Jannis")
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--num_tresh", type=int, default=5, help="Number of thresholds")
    parser.add_argument("--include_trn", action="store_true", help="Include training data")
    parser.add_argument("--include_val", action="store_true", help="Include validation data")
    parser.add_argument("--fraction_only", action="store_true", help="Compute only the fraction of points in the ball")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--precomputed_explanations", action="store_true", help="Use precomputed explanations")
    parser.add_argument("--chunk_size", type=int, default=100, help="Chunk size of test set computed at once")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--kernel_width", type=float, default=None, help="Kernel size for the locality analysis")
    parser.add_argument("--model_regressor", type=str, default="ridge", help="Model regressor for LIME")


    args = parser.parse_args()
    print("Starting the experiment with the following arguments: ", args)
    main(args)