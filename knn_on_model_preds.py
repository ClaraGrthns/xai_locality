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
from src.utils.metrics import binary_classification_metrics
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

def main(args):
    print("Starting the experiment with the following arguments: ", args)
    args.method = ""
    set_random_seeds(args.random_seed)
    results_path = get_path(args.results_folder, args.results_path, args.setting)
    if not osp.exists(results_path):
        os.makedirs(results_path)
    print("saving results to: ", results_path)
    model_handler = ModelHandlerFactory.get_handler(args)
    model = model_handler.model
    _, tst_for_dist, _, _, df_for_expl = model_handler.load_data()

    df_loader = DataLoader(df_for_expl, shuffle=False, batch_size=args.chunk_size)

    ### Create trn set
    # ------ Create trn features
    X_trn = df_for_expl.features
    if isinstance(X_trn, torch.Tensor):
        X_trn = X_trn.numpy()
    # ------ Create trn labels
    ys_trn = []
    print("Computing model predictions on training data")
    with torch.no_grad():  # Disable gradient computation to save memory
        for i, batch in enumerate(tqdm(df_loader, desc="Computing model predictions")):
            preds = model_handler.predict_fn(batch)
            if isinstance(preds, torch.Tensor):
                preds = preds.numpy()
            ys_trn.append(preds)
            if args.debug and i == 10:
                break
    ys_trn = np.concatenate(ys_trn, axis=0)
    if ys_trn.shape[-1] == 1:
        ys_trn_sig = 1 / (1 + np.exp(-ys_trn))
        ys_trn_softmaxed = np.concatenate([1 - ys_trn_sig, ys_trn_sig], axis=-1)
    else: 
        exp_ys_trn = np.exp(ys_trn - np.max(ys_trn, axis=-1, keepdims=True))
        ys_trn_softmaxed = exp_ys_trn / np.sum(exp_ys_trn, axis=-1, keepdims=True)

    ys_trn_label = np.argmax(ys_trn_softmaxed, axis=-1)

    
    #### Create tst set
    #------ Create tst features
    X_tst = tst_for_dist
    if isinstance(X_tst, torch.Tensor):
        X_tst = X_tst.numpy()

    #------ Create tst labels
    with torch.no_grad():
        y_true = model_handler.predict_fn(tst_for_dist)
        if isinstance(preds, torch.Tensor):
            y_true = y_true.numpy()
    if y_true.shape[-1] == 1:
        ys_true_sig = 1 / (1 + np.exp(-y_true))
        ys_true_softmaxed = np.concatenate([1 - ys_true_sig, ys_true_sig], axis=-1)
    else: 
        exp_y_true = np.exp(y_true - np.max(y_true, axis=-1, keepdims=True))
        ys_true_softmaxed = exp_y_true / np.sum(exp_y_true, axis=-1, keepdims=True)
    y_true_label = np.argmax(ys_true_softmaxed, axis=-1)

    #### Experiment Setup
    k_nns = np.arange(args.min_k, args.max_k + 1, args.k_step)
    res = np.zeros((len(k_nns), 4))
    if args.data_folder and args.setting:
        file_name_wo_file_ending = args.setting
    elif args.data_path:
        file_name_wo_file_ending = Path(args.data_path).stem
    else:
        raise ValueError("You must provide either data_folder and setting or data_path.")
    experiment_setting = f"kNN_on_model_preds_{args.model_type}_{file_name_wo_file_ending}_dist_measure-{args.distance_measure}_random_seed-{args.random_seed}"
   

    for i, k_neighbors in enumerate(k_nns):
        print(f"Computing kNN with k={k_neighbors}")
        ## Fit classifier on training data
        classifier = KNeighborsClassifier(n_neighbors=k_neighbors, metric=args.distance_measure)
        classifier.fit(X_trn, ys_trn_label)
        y_pred = classifier.predict(X_tst)
        _, accuracy, precision, recall, f1 = binary_classification_metrics(y_true_label, y_pred, None)
        res[i] = [accuracy, precision, recall, f1]
    print("Best results on accuracy and f1: ", np.max(res[:, 0]), np.max(res[:, 3]))
    print("Best k: ", k_nns[np.argmax(res[:, 0])], k_nns[np.argmax(res[:, 3])])
    # save results and knns with numpy
    res_dict = {"metrics": res, "k_nns": k_nns}
    np.savez(osp.join(results_path, experiment_setting ), **res_dict)


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

    parser.add_argument("--data_path", type=str, help="Path to the data")#, default = "/home/grotehans/xai_locality/data/ExcelFormer_higgs_normalized_data.pt")
    parser.add_argument("--model_path", type=str, help="Path to the model")#, default = "/home/grotehans/xai_locality/pretrained_models/ExcelFormer/higgs/ExcelFormer_normalized_binary_higgs_results.pt")
    parser.add_argument("--results_path", type=str,  help="Path to save results")#,default = "/home/grotehans/xai_locality/results/knn_model_preds/ExcelFormer/higgs/" )
    
    # Model and method configuration
    parser.add_argument("--model_type", type=str, help="Model type: [LightGBM, tab_inception_v3, pt_frame_lgm, pt_frame_xgboost, binary_inception_v3, inception_v3, ExcelFormer, MLP, Trompt]", default = "ExcelFormer")
    
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
    parser.add_argument("--max_k", type=int, default=20)
    parser.add_argument("--k_step", type=int, default=1)
    
    args = parser.parse_args()

    print("Starting the experiment with the following arguments: ", args)
    main(args)
