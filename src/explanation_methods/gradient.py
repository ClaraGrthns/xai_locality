from src.explanation_methods.base import BaseExplanationMethodHandler
from captum.attr import IntegratedGradients, NoiseTunnel
import torch
from src.explanation_methods.gradient_methods.local_classifier import compute_gradmethod_preds_for_all_kNN, compute_saliency_maps
import os.path as osp
import os
import h5py
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import time 
from src.utils.metrics import binary_classification_metrics_per_row, regression_metrics_per_row, impurity_metrics_per_row

class IntegratedGradientsHandler(BaseExplanationMethodHandler):
    def set_explainer(self, **kwargs):
        model = kwargs.get("model")
        self.explainer = IntegratedGradients(model, multiply_by_inputs=False)

    def explain_instance(self, **kwargs):
        return self.explainer.attribute(kwargs["input"], target=kwargs["target"])
    
    
    def compute_explanations(self, results_path, predict_fn, tst_data):
        tst_feat_for_expl_loader = DataLoader(tst_data, batch_size=self.args.chunk_size, shuffle=False)
        device = torch.device("cpu")
        saliency_map_folder = osp.join(results_path, 
                                        "saliency_maps")
        saliency_map_file_path = osp.join(saliency_map_folder, f"saliency_map_{self.args.gradient_method}.h5")
        print("Looking for saliency maps in: ", saliency_map_file_path)
        if osp.exists(saliency_map_file_path):
            print(f"Using precomputed saliency maps from: {saliency_map_file_path}")
            with h5py.File(saliency_map_file_path, "r") as f:
                saliency_maps = f["saliency_map"][:]
            saliency_maps = torch.tensor(saliency_maps).float().to(device)
        else:
            print("Precomputed saliency maps not found. Computing saliency maps for the test set...")
            if not osp.exists(saliency_map_folder):
                os.makedirs(saliency_map_folder)
            saliency_maps = compute_saliency_maps(self.explainer, predict_fn, tst_feat_for_expl_loader)
            with h5py.File(saliency_map_file_path, "w") as f:
                f.create_dataset("saliency_map", data=saliency_maps.cpu().numpy())
        return saliency_maps
    
    def get_experiment_setting(self, fractions):
        return f"fractions-0-{np.round(fractions, 2)}_grad_method-{self.args.gradient_method}_model_type-{self.args.model_type}_dist_measure-{self.args.distance_measure}_accuracy_fraction"      
    
    # def prepare_data_for_analysis(self, dataset, df_feat):
    #     args = self.args
    #     indices = np.random.permutation(len(dataset))
    #     tst_indices, analysis_indices = np.split(indices, [args.max_test_points])
    #     print("using the following indices for testing: ", tst_indices)
    #     tst_data = Subset(dataset, tst_indices)
    #     analysis_data = Subset(dataset, analysis_indices)
    #     tst_feat, analysis_feat = np.split(df_feat[indices], [args.max_test_points])
    #     data_loader_tst = DataLoader(tst_data, batch_size=args.chunk_size, shuffle=False)
    #     return tst_feat, analysis_feat, data_loader_tst, analysis_data

    def iterate_over_data(self, 
                     tst_feat_for_expl, 
                     tst_feat_for_dist, 
                     df_feat_for_expl, 
                     explanations, 
                     n_points_in_ball, 
                     predict_fn, 
                     tree,
                     results_path,
                     experiment_setting,
                     results):
        chunk_size = int(np.min((self.args.chunk_size, len(tst_feat_for_expl))))
        tst_feat_for_expl_loader = DataLoader(tst_feat_for_expl, batch_size=chunk_size, shuffle=False)

        for i, batch in enumerate(tst_feat_for_expl_loader):
            tst_chunk = batch#[0]
            chunk_start = i*chunk_size
            chunk_end = min(chunk_start + chunk_size, len(tst_feat_for_dist))
            print(f"Processing chunk {i}/{(len(tst_feat_for_expl) + chunk_size - 1) // chunk_size}")
            explanations_chunk = explanations[chunk_start:chunk_end]
            with torch.no_grad():
                predictions = predict_fn(tst_chunk) #shape: num test samples x num classes
                predictions_baseline = predict_fn(torch.zeros_like(tst_chunk)) 
                if predictions.shape[-1] == 1:
                    predictions_sm = torch.sigmoid(predictions)
                    predictions_sm = torch.cat([1 - predictions_sm, predictions_sm], dim=-1)
                else:
                    predictions_sm = torch.softmax(predictions, dim=-1)
                top_labels = torch.argmax(predictions_sm, dim=1).tolist()

            proba_output = self.args.model_type in ["LightGBM", "XGBoost", "LightGBM", "pt_frame_xgb"]
            
            chunk_result = compute_gradmethod_preds_for_all_kNN(tst_feat_for_dist[chunk_start:chunk_end], 
                                                                tst_chunk,
                                                                predictions,
                                                                predictions_baseline,
                                                                df_feat_for_expl, 
                                                                explanations_chunk, 
                                                                predict_fn, 
                                                                n_points_in_ball, 
                                                                tree,
                                                                top_labels,
                                                                proba_output
                                                            )
            model_preds_top_label, model_binary_pred_top_label, model_probs_top_label, local_preds, local_binary_pred_top_labels, local_probs_top_label, dist = chunk_result
            for idx in range(n_points_in_ball):
                R = np.max(dist[:, :idx+1], axis=-1)

                aucroc, acc, precision, recall, f1 = binary_classification_metrics_per_row(model_binary_pred_top_label[:, :idx+1], 
                                                                                local_binary_pred_top_labels[:, :idx+1],
                                                                                local_probs_top_label[:, :idx+1]
                                                                                )
                mse, mae, r2 = regression_metrics_per_row(model_preds_top_label[:, :idx+1],
                                                            local_preds[:, :idx+1])
                
                mse_proba, mae_proba, r2_proba = regression_metrics_per_row(model_probs_top_label[:, :idx+1],
                                                local_probs_top_label[:, :idx+1])
                gini, ratio = impurity_metrics_per_row(model_binary_pred_top_label[:, :idx+1])
                variance_preds = np.var(model_preds_top_label[:, :idx+1], axis=1)
                variance_pred_proba = np.var(model_probs_top_label[:, :idx+1], axis=1)

                acc_constant_clf = np.mean(model_binary_pred_top_label[:, :idx+1], axis=1)

                results["accuraccy_constant_clf"][idx, chunk_start:chunk_end] = acc_constant_clf

                results["aucroc"][idx, chunk_start:chunk_end] = aucroc
                results["accuracy"][idx, chunk_start:chunk_end] = acc
                results["precision"][idx, chunk_start:chunk_end] = precision
                results["recall"][idx, chunk_start:chunk_end] = recall
                results["f1"][idx, chunk_start:chunk_end] = f1
                
                results["mse"][idx, chunk_start:chunk_end] = mse
                results["mae"][idx, chunk_start:chunk_end] = mae
                results["r2"][idx, chunk_start:chunk_end] = r2

                results["mse_proba"][idx, chunk_start:chunk_end] = mse_proba
                results["mae_proba"][idx, chunk_start:chunk_end] = mae_proba
                results["r2_proba"][idx, chunk_start:chunk_end] = r2_proba

                results["gini"][idx, chunk_start:chunk_end] = gini
                results["ratio_all_ones"][idx, chunk_start:chunk_end] = ratio
                results["variance_proba"][idx, chunk_start:chunk_end] = variance_pred_proba
                results["variance_logit"][idx, chunk_start:chunk_end] = variance_preds

                results["radius"][idx, chunk_start:chunk_end] = R

            print(f"Processed chunk {i}/{(len(tst_feat_for_expl) + chunk_size - 1) // chunk_size}")    
            np.savez(osp.join(results_path, experiment_setting), **results)
        return results


        
class SmoothGradHandler(IntegratedGradientsHandler):
    def set_explainer(self, **kwargs):
        model = kwargs.get("model")
        multiply_by_inputs = kwargs.get("multiply_by_inputs")
        self.explainer = NoiseTunnel(IntegratedGradients(model, multiply_by_inputs=multiply_by_inputs))
