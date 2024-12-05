from torch_frame.gbdt import XGBoost
from torch_frame.typing import TaskType
from torch_frame.datasets import DataFrameBenchmark
import os.path as osp
import torch
import numpy as np
import argparse
import lime.lime_tabular
import h5py
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
import time
import random
import xgboost
import concurrent.futures
import gc


from utils.lime_local_classifier import compute_lime_accuracy, get_feat_coeff_intercept, compute_fractions, compute_explanations



def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def compute_lime_accuracies(tst_set, dataset, explanations, explainer, predict_fn, thresholds, tree, pred_threshold):
    out = Parallel(n_jobs=-1)(
        delayed(compute_lime_accuracy)(
            tst_set, dataset, explanations, explainer, predict_fn, dist_threshold, tree, pred_threshold
        )
        for dist_threshold in thresholds
    )
    return out

def main(args):
    set_random_seeds(args.random_seed)

    data_path = args.data_path
    model_path = args.model_path
    distance_measure = args.distance_measure
    include_trn = args.include_trn
    include_val = args.include_val
    model_path = args.model_path

    model = XGBoost(task_type=TaskType.BINARY_CLASSIFICATION, num_classes=2 )
    model.load(model_path)

    data_path = "/home/grotehans/pytorch-frame/data/"
    dataset = DataFrameBenchmark(root=data_path, task_type=TaskType.BINARY_CLASSIFICATION,
                                scale='medium', idx=6)
    dataset.materialize()
    dataset = dataset.shuffle()
    train_dataset, val_dataset, test_dataset = dataset.split()
    train_tensor_frame = train_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    tst_feat, tst_y, tst_types = model._to_xgboost_input(test_tensor_frame)
    val_feat, val_y, val_types = model._to_xgboost_input(val_tensor_frame)
    trn_feat, trn_y, trn_types = model._to_xgboost_input(train_tensor_frame)


    df_feat = tst_feat
    df_y = tst_y
    if args.debug:
        tst_feat = tst_feat[:5]
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


    if args.fraction_only:
        setting = f"thresholds-0-{np.round(first_non_zero)}-max{np.round(max)}num_tresh-{num_tresh}_{df_setting}_results_parallel_comp_fraction_points_in_ball.npy"
        start = time.time()
        def compute_fraction_for_threshold(threshold):
            counts = tree.query_radius(tst_feat, r=threshold, count_only=True)
            return counts / df_feat.shape[0]
        fraction_points_in_ball = Parallel(n_jobs=-1)(
            delayed(compute_fraction_for_threshold)(threshold) for threshold in thresholds
        )
        fraction_points_in_ball = np.array(fraction_points_in_ball)
        end = time.time()
        print("spend time: ", end - start)  
        results = {"fraction_points_in_ball": fraction_points_in_ball,
                    "thresholds": thresholds,
                    "time_compute_fractions": end - start,
                    "include_trn": include_trn,
                    "include_val": include_val
                    }
        np.save(osp.join(results_path, setting), results)
        
    else:
        def predict_fn(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            dummy_labels = np.zeros(X.shape[0])
            dtest = xgboost.DMatrix(X, label=dummy_labels,
                                    feature_types=tst_types,
                                    enable_categorical=True)
            pred = model.model.predict(dtest)
            if model.task_type == TaskType.BINARY_CLASSIFICATION:
                pred = np.column_stack((1 - pred, pred))
            return pred

        first_key = next(iter(train_tensor_frame.col_names_dict))
        feature_names = train_tensor_frame.col_names_dict[first_key]
        explainer = lime.lime_tabular.LimeTabularExplainer(trn_feat, 
                                                            feature_names=feature_names, 
                                                            class_names=[0,1], 
                                                            discretize_continuous=True)
        if args.kernel_width is None:
            args.kernel_width = np.round(np.sqrt(trn_feat.shape[1]) * .75, 2) #Default value

        setting = f"thresholds-0-{np.round(first_non_zero)}-max{np.round(max)}num_tresh-{num_tresh}_{df_setting}_kernel_width-{args.kernel_width}_model_regr-{args.model_regressor}_accuracy_fraction_.npy"
        
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
            explanations = np.load(osp.join(args.results_path, f"explanations/explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}.npy"), allow_pickle=True)
        else:
            print("Computing explanations for the test set")
            explanations = compute_explanations(explainer, tst_feat, predict_fn)
            np.save(osp.join(args.results_path, f"explanations/explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}.npy"), explanations)
            print("Finished computing explanations for the test set")

        chunk_size = args.chunk_size
        # Split the test set into chunks
        for i in range(0, len(tst_feat), chunk_size):
            chunk_end = min(i + chunk_size, len(tst_feat))
            tst_chunk = tst_feat[i:chunk_end]
            
            chunk_results = Parallel(n_jobs=-1)(
                delayed(compute_lime_accuracy)(
                    tst_chunk, df_feat, explanations, explainer, predict_fn, dist_threshold, tree
                )
                for dist_threshold in thresholds
            )
            
            # Unpack results directly into the correct positions in the arrays
            for t, (acc, frac, rad, samp, ratio) in enumerate(chunk_results):
                results["accuracy"][t, i:chunk_end] = acc
                results["fraction_points_in_ball"][t, i:chunk_end] = frac
                results["radius"][t, i:chunk_end] = rad
                results["samples_in_ball"][t, i:chunk_end] = samp
                results["ratio_all_ones"][t, i:chunk_end] = ratio
            
            # Save intermediate results
            np.savez(osp.join(results_path, setting), **results)
            print(f"Processed chunk {i//chunk_size + 1}/{(len(tst_feat) + chunk_size - 1)//chunk_size}")

        print("Finished computing LIME accuracy and fraction of points in the ball")

   
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Locality Analyzer")
    parser.add_argument("--data_path", type=str,  help="Path to the data", default="/home/grotehans/xai_locality/data/")
    parser.add_argument("--model_path", type=str, help="Path to the model", default="/home/grotehans/pytorch-frame/benchmark/results/xgboost_binary_medium_6.pt")
    parser.add_argument("--results_path", type=str,  help="Path to save results", default="/home/grotehans/xai_locality/results/XGBoost/Jannis")
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--num_tresh", type=int, default=10, help="Number of thresholds")
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