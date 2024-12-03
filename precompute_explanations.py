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


from utils.lime_local_classifier import compute_explanations



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
    tst_feat, tst_y, tst_types = model._to_xgboost_input(test_tensor_frame)
    trn_feat, trn_y, trn_types = model._to_xgboost_input(train_tensor_frame)


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


    print("Computing explanations for the test set")
    explanations = compute_explanations(explainer, tst_feat, predict_fn)
    np.save(osp.join(args.results_path, "explanations_test_set.npy"), explanations)

   
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Locality Analyzer")
    parser.add_argument("--data_path", type=str,  help="Path to the data", default="/home/grotehans/xai_locality/data/")
    parser.add_argument("--model_path", type=str, help="Path to the model", default="/home/grotehans/pytorch-frame/benchmark/results/xgboost_binary_medium_6.pt")
    parser.add_argument("--results_path", type=str,  help="Path to save results", default="/home/grotehans/xai_locality/results/XGBoost/Jannis")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    print("Starting the experiment with the following arguments: ", args)
    main(args)