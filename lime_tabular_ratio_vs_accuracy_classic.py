from torch_frame.gbdt import XGBoost
from torch_frame.typing import TaskType
from torch_frame.datasets import DataFrameBenchmark
import os.path as osp
import torch
import numpy as np
import argparse

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
# import time to time process
import time
import random

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
    num_tresh = args.num_tresh
    thresholds = np.concatenate((np.array([1e-5]), np.linspace(first_non_zero, max, num_tresh)))
    results_path = args.results_path

    df_setting = "complete_df" if include_trn and include_val else "only_test"
    setting = f"thresholds-0-{np.round(first_non_zero)}-max{np.round(max)}num_tresh-{len(thresholds)}_{df_setting}_results_fraction_points_in_ball.npy"

    fraction_points_in_ball = np.zeros((len(thresholds), len(tst_feat)))
    start = time.time()
    results = {
            "fraction_points_in_ball": fraction_points_in_ball,
            "thresholds": thresholds,
            "include_trn": include_trn,
            "include_val": include_val,
    }

    for idx_tresh, threshold in enumerate(thresholds):
            counts = tree.query_radius(tst_feat, r=threshold, count_only=True)
            fraction_points_in_ball[idx_tresh] = counts / df_feat.shape[0]
            results['fraction_points_in_ball'] = fraction_points_in_ball
            np.save(osp.join(results_path, setting), results)
    end = time.time()
    print("spend time: ", end - start)
    # save numpy array
    results = {
            "fraction_points_in_ball": fraction_points_in_ball,
            "thresholds": thresholds,
            "time_compute_fractions": end - start,
            "include_trn": include_trn,
            "include_val": include_val,
    }

    np.save(osp.join(results_path, setting), results)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Locality Analyzer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--results_path", type=str, required=True, help="Path to save results")
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--num_tresh", type=int, default=200, help="Number of thresholds")
    parser.add_argument("--include_trn", action="store_true", help="Include training data")
    parser.add_argument("--include_val", action="store_true", help="Include validation data")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)