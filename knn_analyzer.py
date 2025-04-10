import argparse
import os
import os.path as osp
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append(osp.join(os.getcwd(), '..'))
from src.utils.misc import set_random_seeds, get_path
from src.model.factory import ModelHandlerFactory
from src.utils.metrics import binary_classification_metrics, regression_metrics
import time

def train_knn_regressors(X_trn, ys_trn, X_tst, k_neighbors, distance_measure):
    """Train KNN regressors for each output dimension."""
    num_classes = ys_trn.shape[1] if ys_trn.ndim > 1 else 1
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

def load_or_compute_predictions(model_handler, data, loader, save_path, prefix="train", debug=False):
    """Load predictions from file or compute them."""
    if osp.exists(save_path):
        print(f"Loading existing {prefix} predictions from {save_path}")
        predictions = np.load(save_path)
    else:
        predictions = []
        print(f"Computing model predictions on {prefix} data")
        with torch.no_grad(): 
            if loader:
                for i, batch in enumerate(tqdm(loader, desc=f"Computing {prefix} predictions")):
                    preds = model_handler.predict_fn(batch)
                    if isinstance(preds, torch.Tensor):
                        preds = preds.numpy()
                    predictions.append(preds)
                predictions = np.concatenate(predictions, axis=0)
            else:
                predictions = model_handler.predict_fn(data)
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.numpy()
        np.save(save_path, predictions)
        print(f"{prefix.capitalize()} predictions saved to {save_path}")
    return predictions

def process_classification_predictions(preds, proba_output=False):
    """Process model predictions for classification tasks."""
    if not proba_output:
        if preds.shape[-1] == 1:
            sig = 1 / (1 + np.exp(-preds))
            softmaxed = np.concatenate([1 - sig, sig], axis=-1)
        else: 
            exp_preds = np.exp(preds - np.max(preds, axis=-1, keepdims=True))
            softmaxed = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
    else:
        if preds.shape[-1] == 1:
            softmaxed = np.concatenate([1 - preds, preds], axis=-1)
        else:
            softmaxed = preds
    
    predicted_labels = np.argmax(softmaxed, axis=-1)
    return softmaxed, predicted_labels

def run_classification_analysis(args, X_trn, X_tst, ys_trn_preds, y_tst_preds, y_trn, y_tst, 
                               k_nns, results_path, file_name_wo_file_ending, distance_measures):
    """Run KNN analysis for classification tasks."""
    proba_output = args.model_type in ["LightGBM", "XGBoost", "pt_frame_xgb", "LogReg"]
    
    ys_trn_softmaxed, ys_trn_predicted_labels = process_classification_predictions(ys_trn_preds, proba_output)
    ys_true_softmaxed, ys_tst_predicted_labels = process_classification_predictions(y_tst_preds, proba_output)
    
    y_tst_proba_top_label = np.max(ys_true_softmaxed, axis=1) if ys_true_softmaxed.ndim > 1 else ys_true_softmaxed
    y_tst_logit_top_label = np.max(y_tst_preds, axis=1) if y_tst_preds.ndim > 1 and y_tst_preds.shape[-1] > 1 else y_tst_preds.flatten()
    
    for distance_measure in distance_measures:
        print(f"\nProcessing with distance measure: {distance_measure}")
        
        
        experiment_setting = f"kNN_on_model_preds_{args.model_type}_{file_name_wo_file_ending}_dist_measure-{distance_measure}_random_seed-{args.random_seed}"
        if osp.exists(osp.join(results_path, experiment_setting + ".npz")) and not args.force_overwrite:
            print(f"Results for the experiment setting {experiment_setting} already exist. Skipping.")
            continue
        
        res_classification = np.zeros((len(k_nns), 4))
        res_proba_regression = np.zeros((len(k_nns), 3))
        res_logit_regression = np.zeros((len(k_nns), 3))
        res_classification_true_labels = np.zeros((len(k_nns), 4))
        
        for i, k_neighbors in enumerate(k_nns):
            print(f"Computing kNN with k={k_neighbors} and distance measure={distance_measure}")
            
            # Classification on model predictions
            classifier = KNeighborsClassifier(n_neighbors=k_neighbors, metric=distance_measure)
            classifier.fit(X_trn, ys_trn_predicted_labels)
            classifier_preds = classifier.predict(X_tst)
            _, accuracy, precision, recall, f1 = binary_classification_metrics(
                ys_tst_predicted_labels, classifier_preds, None)
            res_classification[i] = [accuracy, precision, recall, f1]
            
            # Regression on probabilities
            regressors, regressor_preds = train_knn_regressors(
                X_trn, ys_trn_softmaxed, X_tst, k_neighbors, distance_measure)
            
            if regressor_preds.ndim > 1:
                regressor_preds_top_label = regressor_preds[
                    np.arange(len(ys_tst_predicted_labels)), ys_tst_predicted_labels]
            else:
                regressor_preds_top_label = regressor_preds
                
            mse, mae, r2 = regression_metrics(y_tst_proba_top_label, regressor_preds_top_label)   
            res_proba_regression[i] = [mse, mae, r2]
            
            # Regression on logits (if not using probability output)
            if not proba_output:
                regressors, regressor_logit = train_knn_regressors(
                    X_trn, ys_trn_preds, X_tst, k_neighbors, distance_measure)
                
                if regressor_logit.ndim == 1:
                    regressor_logit_top_label = regressor_logit
                else:
                    regressor_logit_top_label = regressor_logit[
                        np.arange(len(ys_tst_predicted_labels)), ys_tst_predicted_labels]
                
                mse, mae, r2 = regression_metrics(y_tst_logit_top_label, regressor_logit_top_label)   
                res_logit_regression[i] = [mse, mae, r2]
        
            # Classification on true labels
            classifier = KNeighborsClassifier(n_neighbors=k_neighbors, metric=distance_measure)
            classifier.fit(X_trn, y_trn)
            classifier_preds = classifier.predict(X_tst)
            _, accuracy, precision, recall, f1 = binary_classification_metrics(
                y_tst, classifier_preds, None)
            res_classification_true_labels[i] = [accuracy, precision, recall, f1]
        # Save results for this distance measure
        res_dict = {
            "k_nns": k_nns,
            "classification": res_classification,
            "proba_regression": res_proba_regression,
            "logit_regression": res_logit_regression,
            "classification_true_labels": res_classification_true_labels
        }
        np.savez(osp.join(results_path, experiment_setting), **res_dict)
        print(f"Results saved to {osp.join(results_path, experiment_setting)}")

    # Save model performance metrics
    print("Computing metrics for the actual model")
    auroc, accuracy, precision, recall, f1 = binary_classification_metrics(
        y_tst, ys_tst_predicted_labels, y_tst_proba_top_label)
    print(f"Model performance: AUROC={auroc}, Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}")
    res_model = np.array([auroc, accuracy, precision, recall, f1])
    
    model_res = {"classification_model": res_model}
    model_experiment_setting = f"model_performance_{args.model_type}_{file_name_wo_file_ending}_random_seed-{args.random_seed}"
    
    np.savez(osp.join(results_path, model_experiment_setting), **model_res)
    print(f"Model performance results saved to {osp.join(results_path, model_experiment_setting)}")

def run_regression_analysis(args, X_trn, X_tst, ys_trn_preds, y_tst_preds, y_trn, y_tst,
                          k_nns, results_path, file_name_wo_file_ending, distance_measures):
    """Run KNN analysis for regression tasks."""
    for distance_measure in distance_measures:
        print(f"\nProcessing with distance measure: {distance_measure}")
        experiment_setting = f"kNN_regression_on_model_preds_{args.model_type}_{file_name_wo_file_ending}_dist_measure-{distance_measure}_random_seed-{args.random_seed}"
        
        if osp.exists(osp.join(results_path, experiment_setting + ".npz")) and not args.force_overwrite:
            print(f"Results for the experiment setting {experiment_setting} already exist. Skipping.")
            continue
        
        # Initialize results arrays
        res_regression = np.zeros((len(k_nns), 3))
        res_regression_true_y = np.zeros((len(k_nns), 3))
        
        for i, k_neighbors in enumerate(k_nns):
            print(f"Computing kNN with k={k_neighbors} and distance measure={distance_measure}")
            
            # Regression on model predictions
            kNN_regressor = KNeighborsRegressor(n_neighbors=k_neighbors, metric=distance_measure)
            kNN_regressor.fit(X_trn, ys_trn_preds)
            classifier_preds = kNN_regressor.predict(X_tst)
            mse, mae, r2 = regression_metrics(y_tst_preds, classifier_preds)            
            res_regression[i] = [mse, mae, r2]
            
            # Regression on true labels
            kNN_regressor_truey = KNeighborsRegressor(n_neighbors=k_neighbors, metric=distance_measure)
            kNN_regressor_truey.fit(X_trn, y_trn)
            classifier_preds = kNN_regressor_truey.predict(X_tst)
            mse, mae, r2 = regression_metrics(y_tst, classifier_preds)         
            res_regression_true_y[i] = [mse, mae, r2]   
    
        res_dict = {
            "k_nns": k_nns,
            "res_regression": res_regression,
            "res_regression_true_y": res_regression_true_y,
        }
        np.savez(osp.join(results_path, experiment_setting), **res_dict)
        print(f"Results saved to {osp.join(results_path, experiment_setting)}")

    # Save model performance metrics
    print("Computing metrics for the actual model")
    mse, mae, r2 = regression_metrics(y_tst, y_tst_preds)
    print(f"Model performance: MSE={mse}, MAE={mae}, R2={r2}")
    res_model = np.array([mse, mae, r2])
    model_res = {"regression_model": res_model}
    model_experiment_setting = f"model_regression_performance_{args.model_type}_{file_name_wo_file_ending}_random_seed-{args.random_seed}"
    np.savez(osp.join(results_path, model_experiment_setting), **model_res)
    print(f"Model performance results saved to {osp.join(results_path, model_experiment_setting)}")

def main(args):
    print("Starting the experiment with the following arguments: ", args)
    start_time = time.time()
    args.method = ""
    set_random_seeds(args.random_seed)
    results_path = get_path(args.results_folder, args.results_path, args.setting)
    if not osp.exists(results_path):
        os.makedirs(results_path)
    print("saving results to: ", results_path)
    
    model_handler = ModelHandlerFactory.get_handler(args)
    trn_feat, analysis_feat, tst_feat, y_trn, analysis_y, y_tst = model_handler.load_data_for_kNN()

    X_trn = trn_feat.numpy() if isinstance(trn_feat, torch.Tensor) else trn_feat
    X_tst = tst_feat.numpy() if isinstance(tst_feat, torch.Tensor) else tst_feat

    df_loader = DataLoader(trn_feat, shuffle=False, batch_size=args.chunk_size)
    
    ys_trn_preds_path = osp.join(results_path, "ys_trn_preds.npy")
    ys_trn_preds = load_or_compute_predictions(
        model_handler, trn_feat, df_loader, ys_trn_preds_path, "training", args.debug)
    
    ys_tst_preds_path = osp.join(results_path, "ys_tst_preds.npy")
    y_tst_preds = load_or_compute_predictions(
        model_handler, tst_feat, None, ys_tst_preds_path, "test", args.debug)
    
    k_nns = np.arange(args.min_k, args.max_k + 1, args.k_step)
    
    # Get file name for saving results
    if args.data_path:
        print("knn datapath", args.data_path)
        file_name_wo_file_ending = Path(args.data_path).stem
    else:
        raise ValueError("You must provide either data_folder and setting or data_path.")

    distance_measures = args.distance_measures if args.distance_measures else []
    if args.distance_measure and args.distance_measure not in distance_measures:
        distance_measures.append(args.distance_measure)
    
    if not distance_measures:
        distance_measures = ["euclidean"]
    
    print(f"Processing with distance measures: {distance_measures}")
    
    if args.regression:
        run_regression_analysis(
            args, X_trn, X_tst, ys_trn_preds, y_tst_preds, y_trn, y_tst,
            k_nns, results_path, file_name_wo_file_ending, distance_measures
        )
    else:
        run_classification_analysis(
            args, X_trn, X_tst, ys_trn_preds, y_tst_preds, y_trn, y_tst,
            k_nns, results_path, file_name_wo_file_ending, distance_measures
        )
    
    print("time taken: ", (time.time() - start_time)/60, " minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified KNN Locality Analyzer")

    # Data and paths
    parser.add_argument("--data_folder", type=str, help="Path to the data folder")
    parser.add_argument("--model_folder", type=str, help="Path to the model folder")
    parser.add_argument("--results_folder", type=str, help="Path to the results folder")
    parser.add_argument("--setting", type=str, help="Setting of the experiment")

    parser.add_argument("--data_path", 
                        default="/home/grotehans/xai_locality/data/synthetic_data/n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42_normalized_tensor_frame.pt",
                        type=str, help="Path to the data")
    parser.add_argument("--model_path", 
                        default="/home/grotehans/xai_locality/pretrained_models/LightGBM/synthetic_data/LightGBM_n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42.pt",                        
                        type=str, help="Path to the model")
    parser.add_argument("--results_path", 
                        default="/home/grotehans/xai_locality/results/knn_model_preds/LightGBM/synthetic_data/n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42",
                        type=str, help="Path to save results")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, help="Model type: [LightGBM, XGBoost, ExcelFormer, MLP, Trompt]",
                        default="LightGBM")
    parser.add_argument("--regression", action="store_true", help="Run regression analysis instead of classification")
    
    # Analysis type and parameters
    parser.add_argument("--distance_measure", type=str, help="Single distance measure (legacy support)")
    parser.add_argument("--distance_measures", nargs='+', help="List of distance measures to use")
    parser.add_argument("--include_trn", action="store_true", help="Include training data")
    parser.add_argument("--include_val", action="store_true", help="Include validation data")
    parser.add_argument("--force_overwrite", action="store_true", help="force_overwrite overwrite existing results")
    
    # Other parameters
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size of test set computed at once")
    parser.add_argument("--debug", action="store_true", help="Debug")
    parser.add_argument("--min_k", type=int, default=1)
    parser.add_argument("--max_k", type=int, default=25)
    parser.add_argument("--k_step", type=int, default=1)
    
    args = parser.parse_args()
    
    main(args)
