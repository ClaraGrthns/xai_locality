import argparse

import os
import os.path as osp
import sys
from types import SimpleNamespace
from torch.utils.data import DataLoader
import torch
import numpy as np

sys.path.append(osp.join(os.getcwd(), '..'))
from src.utils.misc import set_random_seeds, get_path
from src.model.factory import ModelHandlerFactory
from src.utils.metrics import binary_classification_metrics, regression_metrics
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tqdm import tqdm

def train_knn_regressors(X_trn, ys_trn, X_tst, k_neighbors, distance_measure):
    """Train KNN regressors for each output dimension."""
    num_classes = ys_trn.shape[1]
    regressors = []
    predictions = []
    
    # Train a regressor for each output dimension
    for class_idx in range(num_classes):
        regressor = KNeighborsRegressor(
            n_neighbors=k_neighbors, 
            metric=distance_measure
        )
        # Train on probabilities for current class
        y_trn_class = ys_trn[:, class_idx] if num_classes > 1 else ys_trn
        regressor.fit(X_trn, y_trn_class)
        regressors.append(regressor)
        
        # Predict probabilities for test set
        class_predictions = regressor.predict(X_tst)
        predictions.append(class_predictions)
    
    # Stack predictions into (n_samples, n_classes) array
    predictions = np.stack(predictions, axis=1) if num_classes > 1 else np.array(predictions).flatten()
    
    return regressors, predictions


def main(args):
    print("Starting the experiment with the following arguments: ", args)
    args.method = ""
    set_random_seeds(args.random_seed)
    results_path = get_path(args.results_folder, args.results_path, args.setting)
    if not osp.exists(results_path):
        os.makedirs(results_path)
    print("saving results to: ", results_path)
    model_handler = ModelHandlerFactory.get_handler(args)
    trn_feat, analysis_feat, tst_feat, y_trn, analysis_y, y_tst  = model_handler.load_data_for_kNN()

    df_loader = DataLoader(trn_feat, shuffle=False, batch_size=args.chunk_size)

    ### Create trn set
    # ------ Create trn features
    X_trn = trn_feat
    if isinstance(X_trn, torch.Tensor):
        X_trn = X_trn.numpy()
    # ------ Create trn labels
    ys_trn_preds = []
    print("Computing model predictions on training data")
    with torch.no_grad():  # Disable gradient computation to save memory
        for i, batch in enumerate(tqdm(df_loader, desc="Computing model predictions")):
            preds = model_handler.predict_fn(batch)
            if isinstance(preds, torch.Tensor):
                preds = preds.numpy()
            ys_trn_preds.append(preds)
            if args.debug and i == 10:
                break
    ys_trn_preds = np.concatenate(ys_trn_preds, axis=0)
    proba_output = args.model_type in ["LightGBM", "XGBoost", "pt_frame_lgm", "pt_frame_xgb"]
    if not proba_output:
        if ys_trn_preds.shape[-1] == 1:
            ys_trn_sig = 1 / (1 + np.exp(-ys_trn_preds))
            ys_trn_softmaxed = np.concatenate([1 - ys_trn_sig, ys_trn_sig], axis=-1)
        else: 
            exp_ys_trn = np.exp(ys_trn_preds - np.max(ys_trn_preds, axis=-1, keepdims=True))
            ys_trn_softmaxed = exp_ys_trn / np.sum(exp_ys_trn, axis=-1, keepdims=True)
    else:
        ys_trn_softmaxed = ys_trn_preds
    ys_trn_predicted_labels = np.argmax(ys_trn_softmaxed, axis=-1)
    ys_trn_labels = ys_trn_preds
    
    #### Create tst set
    #------ Create tst features
    X_tst = tst_feat
    if isinstance(X_tst, torch.Tensor):
        X_tst = X_tst.numpy()

    #------ Create tst labels
    with torch.no_grad():
        y_tst_preds = model_handler.predict_fn(tst_feat)
        if isinstance(preds, torch.Tensor):
            y_tst_preds = y_tst_preds.numpy()

    if not proba_output:
        if y_tst_preds.shape[-1] == 1:
            ys_true_sig = 1 / (1 + np.exp(-y_tst_preds))
            ys_true_softmaxed = np.concatenate([1 - ys_true_sig, ys_true_sig], axis=-1)
        else: 
            exp_y_true = np.exp(y_tst_preds - np.max(y_tst_preds, axis=-1, keepdims=True))
            ys_true_softmaxed = exp_y_true / np.sum(exp_y_true, axis=-1, keepdims=True)
    else:
        ys_true_softmaxed = y_tst_preds  
         
    ys_tst_predicted_labels = np.argmax(ys_true_softmaxed, axis=-1)
    y_tst_proba_top_label = np.max(ys_true_softmaxed, axis=1)
    y_tst_logit_top_label = np.max(y_tst_preds, axis=1) if y_tst_preds.shape[-1] > 1 else y_tst_preds.flatten()

    #### Experiment Setup
    k_nns = np.arange(args.min_k, args.max_k + 1, args.k_step)
    res_classification = np.zeros((len(k_nns), 4))
    res_proba_regression = np.zeros((len(k_nns), 3))
    res_logit_regression = np.zeros((len(k_nns), 3))

    res_classification_true_labels = np.zeros((len(k_nns), 4))
    if args.data_folder and args.setting:
        file_name_wo_file_ending = args.setting
    elif args.data_path:
        file_name_wo_file_ending = Path(args.data_path).stem
    else:
        raise ValueError("You must provide either data_folder and setting or data_path.")
    experiment_setting = f"kNN_on_model_preds_{args.model_type}_{file_name_wo_file_ending}_dist_measure-{args.distance_measure}_random_seed-{args.random_seed}"
    # if osp.exists(osp.join(results_path, experiment_setting)):
    #     print(f"Results for the experiment setting {experiment_setting} already exist. Exiting.")
    #     return
    
    # train kNN on the model predictions
    for i, k_neighbors in enumerate(k_nns):
        print(f"Computing kNN with k={k_neighbors}")
        ## Fit classifier on training data
        classifier = KNeighborsClassifier(n_neighbors=k_neighbors, metric=args.distance_measure)
        classifier.fit(X_trn, ys_trn_predicted_labels)
        classifier_preds = classifier.predict(X_tst)
        _, accuracy, precision, recall, f1 = binary_classification_metrics(ys_tst_predicted_labels, classifier_preds, None)
        res_classification[i] = [accuracy, precision, recall, f1]

        ## Fit regressor on training data
        regressors, regressor_preds = train_knn_regressors(
                                        X_trn, 
                                        ys_trn_softmaxed,
                                        X_tst, 
                                        k_neighbors,
                                        args.distance_measure
                                    )
        regressor_preds_top_label = regressor_preds[np.arange(len(ys_tst_predicted_labels)), ys_tst_predicted_labels]
        mse, mae, r2 = regression_metrics(y_tst_proba_top_label, regressor_preds_top_label)   
        res_proba_regression[i] = [mse, mae, r2]

        if not proba_output:
             ## Fit regressor on training data
            regressors, regressor_logit = train_knn_regressors(
                                            X_trn, 
                                            ys_trn_preds,
                                            X_tst, 
                                            k_neighbors,
                                            args.distance_measure
                                        )
            if regressor_logit.ndim == 1:
                regressor_logit_top_label = regressor_logit
            else:
                regressor_logit_top_label = regressor_logit[np.arange(len(ys_tst_predicted_labels)), ys_tst_predicted_labels]
            mse, mae, r2 = regression_metrics(y_tst_logit_top_label, regressor_logit_top_label)   
            res_logit_regression[i] = [mse, mae, r2]  
    
    # train kNN on the actual trn labels
    for i, k_neighbors in enumerate(k_nns):
        print(f"Computing kNN with k={k_neighbors}")
        ## Fit classifier on training data
        classifier = KNeighborsClassifier(n_neighbors=k_neighbors, metric=args.distance_measure)
        classifier.fit(X_trn, y_trn)
        classifier_preds = classifier.predict(X_tst)
        _, accuracy, precision, recall, f1 = binary_classification_metrics(ys_tst_predicted_labels, y_tst, None)
        res_classification_true_labels[i] = [accuracy, precision, recall, f1]


    res_dict = {
        "k_nns": k_nns,
        "classification": res_classification,
        "proba_regression": res_proba_regression,
        "logit_regression": res_logit_regression,
        "classification_true_labels": res_classification_true_labels
    }
    np.savez(osp.join(results_path, experiment_setting ), **res_dict)
    print(f"Results saved to {osp.join(results_path, experiment_setting)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locality Analyzer")

#   results_path: "/home/grotehans/xai_locality/results/gradient_methods/integrated_gradient/ExcelFormer/higgs"
#   data_path: "/home/grotehans/xai_locality/data/ExcelFormer_higgs_normalized_data.pt"
#   model_path: "/home/grotehans/xai_locality/pretrained_models/ExcelFormer/higgs/ExcelFormer_normalized_binary_higgs_results.pt"
    # Data and model paths
    parser.add_argument("--data_folder", type=str, help="Path to the data folder")
    parser.add_argument("--model_folder", type=str, help="Path to the model folder")
    parser.add_argument("--results_folder", type=str,help="Path to the results folder" )
    parser.add_argument("--setting", type=str, help="Setting of the experiment")

    parser.add_argument("--data_path", 
                        default = "/home/grotehans/xai_locality/data/synthetic_data/n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42.npz",
                        type=str, help="Path to the data")#, default = "/home/grotehans/xai_locality/data/ExcelFormer_higgs_normalized_data.pt")
    parser.add_argument("--model_path", 
                        default = "/home/grotehans/xai_locality/pretrained_models/LightGBM/synthetic_data/LightGBM_n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42",                        
                        type=str, help="Path to the model")#, default = "/home/grotehans/xai_locality/pretrained_models/ExcelFormer/higgs/ExcelFormer_normalized_binary_higgs_results.pt")
    parser.add_argument("--results_path", 
                        default= "/home/grotehans/xai_locality/results/knn_model_preds/LightGBM/synthetic_data/n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42",
                        type=str,  help="Path to save results")#,default = "/home/grotehans/xai_locality/results/knn_model_preds/ExcelFormer/higgs/" )
    
    # Model and method configuration
    parser.add_argument("--model_type", type=str, help="Model type: [LightGBM, tab_inception_v3, pt_frame_lgm, pt_frame_xgboost, binary_inception_v3, inception_v3, ExcelFormer, MLP, Trompt]",
                        default = "LightGBM",
                        )
    
    # Analysis parameters
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--include_trn", action="store_true", help="Include training data")
    parser.add_argument("--include_val", action="store_true", help="Include validation data")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk_size", type=int, help="Chunk size of test set computed at once", default=10)
    parser.add_argument("--debug", action="store_true", help="Debug")
    
    # Other parameters
    parser.add_argument("--max_test_points", type=int, default=200)
    parser.add_argument("--min_k", type=int, default=1)
    parser.add_argument("--max_k", type=int, default=25)
    parser.add_argument("--k_step", type=int, default=1)
    
    args = parser.parse_args()

    print("Starting the experiment with the following arguments: ", args)
    main(args)
