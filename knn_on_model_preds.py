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
# Start tracking time
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

def main(args):
    print("Starting the experiment with the following arguments: ", args)
    start_time = time.time()
    args.method = ""
    set_random_seeds(args.random_seed)
    results_path = get_path(args.results_folder, args.results_path, args.setting)
    if not osp.exists(results_path):
        os.makedirs(results_path)
    print("saving results to: ", results_path)
    
    # Initialize the model handler
    model_handler = ModelHandlerFactory.get_handler(args)
    trn_feat, analysis_feat, tst_feat, y_trn, analysis_y, y_tst = model_handler.load_data_for_kNN()

    # Convert to numpy if needed
    X_trn = trn_feat.numpy() if isinstance(trn_feat, torch.Tensor) else trn_feat
    X_tst = tst_feat.numpy() if isinstance(tst_feat, torch.Tensor) else tst_feat

    # For model predictions on training data (if analyzing model predictions)
    # Set up DataLoader for batch processing
    df_loader = DataLoader(trn_feat, shuffle=False, batch_size=args.chunk_size)
    
    # Load or compute training predictions
    ys_trn_preds_path = osp.join(results_path, "ys_trn_preds.npy")
    if osp.exists(ys_trn_preds_path):
        print(f"Loading existing training labels from {ys_trn_preds_path}")
        ys_trn_preds = np.load(ys_trn_preds_path)
    else:
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
        np.save(ys_trn_preds_path, ys_trn_preds)
        print(f"Training labels saved to {ys_trn_preds_path}")
    
    # Process the model predictions
    proba_output = args.model_type in ["LightGBM", "XGBoost", "pt_frame_xgb", "LogisticRegression"]
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

    # Load or compute test predictions
    ys_tst_preds_path = osp.join(results_path, "ys_tst_preds.npy")
    if osp.exists(ys_tst_preds_path):
        print(f"Loading existing test labels from {ys_tst_preds_path}")
        y_tst_preds = np.load(ys_tst_preds_path)
    else:
        print("Computing model predictions on test data")
        with torch.no_grad():
            y_tst_preds = model_handler.predict_fn(tst_feat)
            if isinstance(y_tst_preds, torch.Tensor):
                y_tst_preds = y_tst_preds.numpy()
        np.save(ys_tst_preds_path, y_tst_preds)
        print(f"Test labels saved to {ys_tst_preds_path}")

    # Process test predictions
    proba_output = args.model_type in ["LightGBM", "XGBoost", "pt_frame_xgb", "LogisticRegression"]
    if not proba_output:
        if y_tst_preds.shape[-1] == 1:
            ys_true_sig = 1 / (1 + np.exp(-y_tst_preds))
            ys_true_softmaxed = np.concatenate([1 - ys_true_sig, ys_true_sig], axis=-1)
        else: 
            exp_y_true = np.exp(y_tst_preds - np.max(y_tst_preds, axis=-1, keepdims=True))
            ys_true_softmaxed = exp_y_true / np.sum(exp_y_true, axis=-1, keepdims=True)
    else:
        if y_tst_preds.shape[-1] == 1:
            ys_true_softmaxed = np.concatenate([1 - y_tst_preds, y_tst_preds], axis=-1)
        else:
            ys_true_softmaxed = y_tst_preds
         
    ys_tst_predicted_labels = np.argmax(ys_true_softmaxed, axis=-1)
    
    # For regression analysis (if analyzing model predictions)
    y_tst_proba_top_label = np.max(ys_true_softmaxed, axis=1) if ys_true_softmaxed.ndim > 1 else ys_true_softmaxed
    y_tst_logit_top_label = np.max(y_tst_preds, axis=1) if y_tst_preds.ndim > 1 and y_tst_preds.shape[-1] > 1 else y_tst_preds.flatten()

    # Define k values to test
    k_nns = np.arange(args.min_k, args.max_k + 1, args.k_step)
    
    # Get file name for saving results
    if args.data_folder and args.setting:
        file_name_wo_file_ending = args.setting
    elif args.data_path:
        file_name_wo_file_ending = Path(args.data_path).stem
    else:
        raise ValueError("You must provide either data_folder and setting or data_path.")

    # Process all distance measures
    distance_measures = args.distance_measures if args.distance_measures else []
    if args.distance_measure and args.distance_measure not in distance_measures:
        distance_measures.append(args.distance_measure)
    
    # If no distance measures are provided, use euclidean as default
    if not distance_measures:
        distance_measures = ["euclidean"]
    
    print(f"Processing with distance measures: {distance_measures}")
    
    # If analyzing model predictions, set up result storage
    res_classification = np.zeros((len(k_nns), 4))
    res_proba_regression = np.zeros((len(k_nns), 3))
    res_logit_regression = np.zeros((len(k_nns), 3))
    
    # Results storage for true label analysis
    res_classification_true_labels = np.zeros((len(k_nns), 4))
    
    # # Process each distance measure
    # for distance_measure in distance_measures:
    #     print(f"\nProcessing with distance measure: {distance_measure}")
        
    #     # Define result filename
    #     experiment_setting = f"kNN_on_model_preds_{args.model_type}_{file_name_wo_file_ending}_dist_measure-{distance_measure}_random_seed-{args.random_seed}"
        
    #     # # Skip if results already exist and not forcing overwrite
    #     # if osp.exists(osp.join(results_path, experiment_setting + ".npz")) and not args.force_overwrite:
    #     #     print(f"Results for the experiment setting {experiment_setting} already exist. Skipping.")
    #     #     continue
        
    #     # Process each k value
    #     for i, k_neighbors in enumerate(k_nns):
    #         print(f"Computing kNN with k={k_neighbors} and distance measure={distance_measure}")
            
    #         # If analyzing model predictions
    #         classifier = KNeighborsClassifier(n_neighbors=k_neighbors, metric=distance_measure)
    #         classifier.fit(X_trn, ys_trn_predicted_labels)
    #         classifier_preds = classifier.predict(X_tst)
    #         _, accuracy, precision, recall, f1 = binary_classification_metrics(ys_tst_predicted_labels, classifier_preds, None)
    #         res_classification[i] = [accuracy, precision, recall, f1]
            
    #         # Regression on probabilities
    #         regressors, regressor_preds = train_knn_regressors(
    #                                         X_trn, 
    #                                         ys_trn_softmaxed,
    #                                         X_tst, 
    #                                         k_neighbors,
    #                                         distance_measure
    #                                     )
    #         if regressor_preds.ndim > 1:
    #             regressor_preds_top_label = regressor_preds[np.arange(len(ys_tst_predicted_labels)), ys_tst_predicted_labels]
    #         else:
    #             regressor_preds_top_label = regressor_preds
                
    #         mse, mae, r2 = regression_metrics(y_tst_proba_top_label, regressor_preds_top_label)   
    #         res_proba_regression[i] = [mse, mae, r2]
            
    #         # Regression on logits (if not using probability output)
    #         if not proba_output:
    #             regressors, regressor_logit = train_knn_regressors(
    #                                             X_trn, 
    #                                             ys_trn_preds,
    #                                             X_tst, 
    #                                             k_neighbors,
    #                                             distance_measure
    #                                         )
    #             if regressor_logit.ndim == 1:
    #                 regressor_logit_top_label = regressor_logit
    #             else:
    #                 regressor_logit_top_label = regressor_logit[np.arange(len(ys_tst_predicted_labels)), ys_tst_predicted_labels]
    #             mse, mae, r2 = regression_metrics(y_tst_logit_top_label, regressor_logit_top_label)   
    #             res_logit_regression[i] = [mse, mae, r2]
        
    #         # Classification on true labels (always done)
    #         classifier = KNeighborsClassifier(n_neighbors=k_neighbors, metric=distance_measure)
    #         classifier.fit(X_trn, y_trn)
    #         classifier_preds = classifier.predict(X_tst)
    #         _, accuracy, precision, recall, f1 = binary_classification_metrics(y_tst, classifier_preds, None)
    #         res_classification_true_labels[i] = [accuracy, precision, recall, f1]
    
    #     # Save results for this distance measure
    #     res_dict = {
    #         "k_nns": k_nns,
    #         "classification": res_classification,
    #         "proba_regression": res_proba_regression,
    #         "logit_regression": res_logit_regression,
    #         "classification_true_labels": res_classification_true_labels
    #     }
    #     np.savez(osp.join(results_path, experiment_setting), **res_dict)
    #     print(f"Results saved to {osp.join(results_path, experiment_setting)}")

    # Save model performance metrics (only needs to be done once)
    print("Computing metrics for the actual model")
    auroc, accuracy, precision, recall, f1 = binary_classification_metrics(y_tst, ys_tst_predicted_labels, y_tst_proba_top_label)
    res_model = np.array([auroc, accuracy, precision, recall, f1])
    
    model_res = {"classification_model": res_model}
    model_experiment_setting = f"model_performance_{args.model_type}_{file_name_wo_file_ending}_random_seed-{args.random_seed}"
    
    np.savez(osp.join(results_path, model_experiment_setting), **model_res)
    print(f"Model performance results saved to {osp.join(results_path, model_experiment_setting)}")
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
    
    # Analysis type and parameters
    parser.add_argument("--analyze_model_preds", action="store_true", help="Analyze model predictions (like knn_on_model_preds.py)")
    parser.add_argument("--distance_measure", type=str, help="Single distance measure (legacy support)")
    parser.add_argument("--distance_measures", nargs='+', help="List of distance measures to use")
    parser.add_argument("--include_trn", action="store_true", help="Include training data")
    parser.add_argument("--include_val", action="store_true", help="Include validation data")
    parser.add_argument("--force_overwrite", action="store_true", help="Force overwrite existing results")
    
    # Other parameters
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size of test set computed at once")
    parser.add_argument("--debug", action="store_true", help="Debug")
    parser.add_argument("--use_true_labels", action="store_true", help="Use true labels for kNN")
    parser.add_argument("--max_test_points", type=int, default=200)
    parser.add_argument("--min_k", type=int, default=1)
    parser.add_argument("--max_k", type=int, default=25)
    parser.add_argument("--k_step", type=int, default=1)
    
    args = parser.parse_args()
    
    main(args)
