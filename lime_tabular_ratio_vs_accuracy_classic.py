from torch_frame.gbdt import XGBoost
from torch_frame.typing import TaskType
from torch_frame.datasets import DataFrameBenchmark
import os.path as osp
import torch
import numpy as np
import argparse
import lime.lime_tabular
import xgboost
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
# import time to time process
import time
import random
from utils.lime_local_classifier import compute_lime_accuracy

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
    val_feat, val_y, _ = model._to_xgboost_input(val_tensor_frame)
    trn_feat, trn_y, _ = model._to_xgboost_input(train_tensor_frame)


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
    # setting = f"thresholds-0-{np.round(first_non_zero)}-max{np.round(max)}num_tresh-{len(thresholds)}_{df_setting}_results_fraction_points_in_ball.npy"
    setting = f"thresholds-0-{np.round(first_non_zero)}-max{np.round(max)}num_tresh-{len(thresholds)}_{df_setting}_results_complete.npy"


    first_key = next(iter(train_tensor_frame.col_names_dict))
    feature_names = train_tensor_frame.col_names_dict[first_key]
    explainer = lime.lime_tabular.LimeTabularExplainer(trn_feat, 
                                                        feature_names=feature_names, 
                                                        class_names=[0,1], 
                                                        discretize_continuous=True)

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

    start = time.time()
    # fraction_points_in_ball = np.zeros((len(thresholds), len(tst_feat)))
    # results = {
    #         "fraction_points_in_ball": fraction_points_in_ball,
    #         "thresholds": thresholds,
    #         "include_trn": include_trn,
    #         "include_val": include_val,
    # }

    # for idx_tresh, threshold in enumerate(thresholds):
    #         counts = tree.query_radius(tst_feat, r=threshold, count_only=True)
    #         fraction_points_in_ball[idx_tresh] = counts / df_feat.shape[0]
    #         results['fraction_points_in_ball'] = fraction_points_in_ball
    #         np.save(osp.join(results_path, setting), results)
    print("start computing")
    print("saving results under: ", setting)
    results = {key:[] for key in ["accuracy", "fraction_points_in_ball", "radius", "samples_in_ball", "ratio_all_ones"]}
    for dp in range(0, len(tst_feat)):
        for threshold in thresholds:
            (accuracy, fraction_points_in_ball, radius, samples_in_ball, ratio_all_ones) = compute_lime_accuracy(
                                        x=tst_feat[dp],
                                        tree=tree,
                                        dataset=df_feat,
                                        explainer=explainer,
                                        predict_fn=predict_fn,
                                        dist_measure=distance_measure,
                                        dist_threshold=threshold)
            results["accuracy"].append(accuracy)
            results["fraction_points_in_ball"].append(fraction_points_in_ball)
            results["radius"].append(radius)
            results["samples_in_ball"].append(samples_in_ball)
            results["ratio_all_ones"].append(ratio_all_ones)
            results['time_compute_fractions'] = time.time() - start
            np.save(osp.join(results_path, setting), results)
        print(f"saved results for datapoints: {dp}/{len(tst_feat)}")

    print("spend time: ", time.time() - start)
    # save numpy array
    # results = {
    #         "fraction_points_in_ball": fraction_points_in_ball,
    #         "thresholds": thresholds,
    #         "time_compute_fractions": time.time() - start,
    #         "include_trn": include_trn,
    #         "include_val": include_val,
    # }

    np.save(osp.join(results_path, setting), results)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Locality Analyzer")
    parser.add_argument("--data_path", type=str,  help="Path to the data", default="/home/grotehans/xai_locality/data/")
    parser.add_argument("--model_path", type=str, help="Path to the model", default="/home/grotehans/pytorch-frame/benchmark/results/xgboost_binary_medium_6.pt")
    parser.add_argument("--results_path", type=str,  help="Path to save results", default="/home/grotehans/xai_locality/results/XGBoost/Jannis")
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--num_tresh", type=int, default=200, help="Number of thresholds")
    parser.add_argument("--include_trn", action="store_true", help="Include training data")
    parser.add_argument("--include_val", action="store_true", help="Include validation data")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)