import argparse
import os
import os.path as osp
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from torch.utils.data import DataLoader
from tqdm import tqdm
# Start tracking time
sys.path.append(osp.join(os.getcwd(), '..'))
from src.utils.misc import set_random_seeds, get_path
from src.model.factory import ModelHandlerFactory
from src.utils.metrics import regression_metrics
import time

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
    
    k_nns = np.arange(args.min_k, args.max_k + 1, args.k_step)
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
    
    res_regression = np.zeros((len(k_nns), 3))
    res_regression_true_y = np.zeros((len(k_nns), 3))
    
    for distance_measure in distance_measures:
        print(f"\nProcessing with distance measure: {distance_measure}")
        experiment_setting = f"kNN_regression_on_model_preds_{args.model_type}_{file_name_wo_file_ending}_dist_measure-{distance_measure}_random_seed-{args.random_seed}"
        if osp.exists(osp.join(results_path, experiment_setting + ".npz")) and not args.force_overwrite:
            print(f"Results for the experiment setting {experiment_setting} already exist. Skipping.")
            continue
        for i, k_neighbors in enumerate(k_nns):
            print(f"Computing kNN with k={k_neighbors} and distance measure={distance_measure}")
            kNN_regressor = KNeighborsRegressor(n_neighbors=k_neighbors, metric=distance_measure)
            kNN_regressor.fit(X_trn, ys_trn_preds)
            classifier_preds = kNN_regressor.predict(X_tst)
            mse, mae, r2 = regression_metrics(y_tst_preds, classifier_preds)            
            res_regression[i] = [mse, mae, r2]
            
            kNN_regressor_truey = KNeighborsRegressor(n_neighbors=k_neighbors, metric=distance_measure)
            kNN_regressor_truey.fit(X_trn, y_trn)
            classifier_preds = kNN_regressor_truey.predict(X_tst)
            mse, mae, r2 = regression_metrics(y_tst_preds, y_tst)         
            res_regression_true_y[i] = [mse, mae, r2]   
    
        res_dict = {
            "k_nns": k_nns,
            "res_regression": res_regression,
            "res_regression_true_y": res_regression_true_y,
        }
        np.savez(osp.join(results_path, experiment_setting), **res_dict)
        print(f"Results saved to {osp.join(results_path, experiment_setting)}")

    # Save model performance metrics (only needs to be done once)
    print("Computing metrics for the actual model")
    mse, mae, r2 = regression_metrics(y_tst, y_tst_preds)
    res_model = np.array([mse, mae, r2])
    model_res = {"regression_model": res_model}
    model_experiment_setting = f"model_regression_performance_{args.model_type}_{file_name_wo_file_ending}_random_seed-{args.random_seed}"
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
    parser.add_argument("--min_k", type=int, default=1)
    parser.add_argument("--max_k", type=int, default=25)
    parser.add_argument("--k_step", type=int, default=1)
    
    args = parser.parse_args()
    
    main(args)
