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
from utils.misc import get_non_zero_cols, set_random_seeds
from lime_analysis.lime_local_classifier import compute_lime_accuracy, compute_explanations
import os


def main(args):
    set_random_seeds(args.random_seed)

    distance_measure = args.distance_measure
    include_trn = args.include_trn
    include_val = args.include_val

    def get_path(base_folder, base_path, setting, suffix=""):
        if base_folder is None:
            return base_path
        assert setting is not None, "Setting must be specified if folder is provided"
        return osp.join(base_folder, f"{suffix}{setting}")

    # Replace the original code with:
    data_path = get_path(args.data_folder, args.data_path, args.setting)
    model_path = get_path(args.model_folder, args.model_path, args.setting, suffix="final_model_")
    results_path = get_path(args.results_folder, args.results_path, args.setting)

    # Add .npz extension specifically for data_path
    if args.data_folder is not None:
        data_path += ".npz"
    
    if args.model_type == "xgboost":
        from model.pytorch_frame_xgboost import load_model, load_data, predict_fn
    elif args.model_type == "lightgbm" and ("synthetic" in results_path):
        from model.lightgbm import load_model, load_data, predict_fn
    elif args.model_type == "lightgbm":
        from model.pytorch_frame_lgm import load_model, load_data, predict_fn
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    model = load_model(model_path)
    tst_feat, tst_y, val_feat, val_y, trn_feat, trn_y = load_data(model, data_path)
    feature_names = np.arange(trn_feat.shape[1])
    predict_fn_wrapped = partial(predict_fn, model=model)
    model_predictions = np.argmax(predict_fn_wrapped(tst_feat), axis=1)

    print("train, test, val feature shapes: ", trn_feat.shape, tst_feat.shape, val_feat.shape)
    df_feat = tst_feat
    df_y = tst_y
    if args.debug:
        tst_feat = tst_feat[:10]
        print("Debug mode: Using only the first 10 samples")

    if include_trn:
        df_feat = np.concatenate([trn_feat, df_feat], axis=0)
        df_y = np.concatenate([trn_y, df_y], axis=0)
    if include_val:
        df_feat = np.concatenate([df_feat, val_feat], axis=0)
        df_y = np.concatenate([df_y, val_y], axis=0)

    # Sample 5000 points from each dataset to estimate min/max distances
    tst_sample_idx = np.random.choice(len(tst_feat), min(5000, len(tst_feat)), replace=False)
    df_sample_idx = np.random.choice(len(df_feat), min(5000, len(df_feat)), replace=False)
   
    tst_sample = tst_feat[tst_sample_idx]
    df_sample = df_feat[df_sample_idx]
    
    # Compute pairwise distances on samples
    distances_pw = pairwise_distances(tst_sample, df_sample, metric=distance_measure)
    max = np.max(distances_pw)
    first_non_zero = np.min(distances_pw[np.round(distances_pw, 2) > 0])

    tree = BallTree(df_feat)  
    num_tresh = args.num_tresh if not args.debug else 2
    thresholds = np.concatenate((np.array([1e-5]), np.linspace(first_non_zero, max, num_tresh)))
    if args.kernel_width is None:
        args.kernel_width = np.round(np.sqrt(trn_feat.shape[1]) * .75, 2) #Default value
    df_setting = "complete_df" if include_trn and include_val else "only_test"
    experiment_setting = f"thresholds-0-{np.round(first_non_zero)}-max{np.round(max)}num_tresh-{num_tresh}_{df_setting}_kernel_width-{args.kernel_width}_model_regr-{args.model_regressor}_model_type-{args.model_type}_accuracy_fraction.npy"
        
    explainer = lime.lime_tabular.LimeTabularExplainer(trn_feat, 
                                                        feature_names=feature_names, 
                                                        class_names=[0,1], #TODO: make this more general
                                                        discretize_continuous=True,
                                                        random_state=args.random_seed,
                                                        kernel_width=args.kernel_width)
    
    if args.debug:
        experiment_setting = f"debug_{experiment_setting}"
        args.precomputed_explanations = False
        print("Start computing LIME accuracy and fraction of points in the ball")
    print("saving results to: ", results_path)

    if not osp.exists(results_path):
        os.makedirs(results_path)
    if args.precomputed_explanations:
        print("Using precomputed explanations")
        explanations = np.load(osp.join(results_path, f"explanations/normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}.npy"), allow_pickle=True)
        print(len(explanations), "explanations loaded")
    else:
        print("Computing explanations for the test set")
        explanations = compute_explanations(explainer, tst_feat, predict_fn_wrapped)
        explanations_dir = osp.join(results_path, "explanations")
        print("explanations_dir: ", explanations_dir)
        if not osp.exists(explanations_dir):
            os.makedirs(explanations_dir)
        if not osp.exists(osp.join(explanations_dir, f"normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}.npy")):
            np.save(osp.join(explanations_dir, f"normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}.npy"), explanations)
        print("Finished computing explanations for the test set")
    
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
    chunk_size = args.chunk_size

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
        for threshold, acc, frac, rad, samp, ratio in chunk_results_sorted:
            threshold_idx = np.where(thresholds == threshold)[0][0]
            results["accuracy"][threshold_idx, i:chunk_end] = acc
            results["fraction_points_in_ball"][threshold_idx, i:chunk_end] = frac
            results["radius"][threshold_idx, i:chunk_end] = rad
            results["samples_in_ball"][threshold_idx, i:chunk_end] = samp
            results["ratio_all_ones"][threshold_idx, i:chunk_end] = ratio
        
        # create graphs for the accuracy and fraction of points in the ball
        graphics_dir = osp.join(results_path, "graphics")
        if not osp.exists(graphics_dir):
            os.makedirs(graphics_dir)
        non_zero_cols = get_non_zero_cols(results["accuracy"])
        plot_accuracy_vs_threshold(accuracy=results["accuracy"][:, :non_zero_cols], 
                                   thresholds=results["thresholds"], 
                                   model_predictions=model_predictions[:non_zero_cols], 
                                   save_path=osp.join(graphics_dir, f"accuracy_vs_threshold_kernel{args.kernel_width}.pdf"))
        plot_accuracy_vs_fraction(accuracy=results["accuracy"][:, :non_zero_cols], 
                                fraction_points_in_ball=results["fraction_points_in_ball"][:, :non_zero_cols], 
                                model_predictions=model_predictions[:non_zero_cols], 
                                save_path=osp.join(graphics_dir, f"accuracy_vs_fraction_kernel{args.kernel_width}.pdf"))
        plot_3d_scatter(fraction_points_in_ball=results["fraction_points_in_ball"][:, :non_zero_cols], 
                        thresholds=results["thresholds"], 
                        accuracy=results["accuracy"][:, :non_zero_cols], 
                        angles=(10, 30), 
                        cmap="tab20b",
                        save_path=osp.join(graphics_dir, f"3d_scatter_kernel{args.kernel_width}.pdf"))
        
        np.savez(osp.join(results_path, experiment_setting), **results)
        print(f"Processed chunk {i//chunk_size + 1}/{(len(tst_feat) + chunk_size - 1)//chunk_size}")
    print("Finished computing LIME accuracy and fraction of points in the ball")

   
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Locality Analyzer")
    parser.add_argument("--data_folder", type=str, help="Path to the data folder")
    parser.add_argument("--model_folder", type=str, help="Path to the model folder")
    parser.add_argument("--results_folder", type=str, help="Path to the results folder")
    parser.add_argument("--setting", type=str, help="Setting of the experiment")
    parser.add_argument("--data_path", type=str, help="Path to the data", default = "/home/grotehans/xai_locality/data/LightGBM_medium_6_normalized_data.pt")
    parser.add_argument("--model_path", type=str, help="Path to the model", default = "/home/grotehans/xai_locality/pretrained_models/lightgbm/jannis/lightgbm_normalized_binary_medium_6.pt")
    parser.add_argument("--model_type", type=str, default="lightgbm", help="Model type, so far only 'xgboost' and 'lightgbm' is supported")
    parser.add_argument("--results_path", type=str, help="Path to save results", default="/home/grotehans/xai_locality/results/lightgbm/jannis")
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--num_tresh", type=int, default=150, help="Number of thresholds")
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

    # Validate arguments
    if (args.data_folder and args.setting and args.model_folder and args.results_folder) or (args.data_path and args.model_path and args.results_path):
        print("Starting the experiment with the following arguments: ", args)
        main(args)
    else:
        parser.error("You must provide either data_folder, model_folder, results_folder, and setting or data_path, model_path, and results_path.")