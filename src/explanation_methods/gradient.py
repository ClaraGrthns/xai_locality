from src.explanation_methods.base import BaseExplanationMethodHandler
from captum.attr import IntegratedGradients, NoiseTunnel
import torch
from src.explanation_methods.gradient_methods.local_classifier import (compute_gradmethod_preds_for_all_kNN, 
                                                                       compute_gradmethod_regressionpreds_for_all_kNN, 
                                                                       compute_saliency_maps)
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
        if osp.exists(saliency_map_file_path) and not self.args.force:
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
        setting = f"fractions-0-{np.round(fractions, 2)}_grad_method-{self.args.gradient_method}_model_type-{self.args.model_type}_dist_measure-{self.args.distance_measure}_accuracy_fraction"      
        if self.args.regression:
            setting = "regression_" + setting
        return setting


    def process_chunk(self, batch, tst_chunk_dist, df_feat_for_expl, explanations_chunk, predict_fn, n_points_in_ball, tree):
        """
        Process a single chunk of data for gradient-based methods.
        """
        tst_chunk = batch  # For gradient methods, batch is already in the right format
        proba_output = self.args.model_type in ["LightGBM", "XGBoost", "LightGBM", "pt_frame_xgb", "LogReg"]
        
        with torch.no_grad():
            predictions = predict_fn(tst_chunk)
            predictions_baseline = predict_fn(torch.zeros_like(tst_chunk))
        if self.args.regression:
            return compute_gradmethod_regressionpreds_for_all_kNN(
                tst_feat = tst_chunk_dist,
                tst_chunk = tst_chunk,
                predictions = predictions,
                predictions_baseline = predictions_baseline,
                analysis_data = df_feat_for_expl, 
                saliency_map = explanations_chunk, 
                predict_fn = predict_fn, 
                n_closest = n_points_in_ball, 
                tree = tree,
            )
        else:
            if not proba_output:
                if predictions.shape[-1] == 1:
                    predictions_sm = torch.sigmoid(predictions)
                    predictions_sm = torch.cat([1 - predictions_sm, predictions_sm], dim=-1)
                else:
                    predictions_sm = torch.softmax(predictions, dim=-1)
            else:
                if predictions.shape[-1] == 1:
                    predictions_sm = torch.cat([1 - predictions, predictions], dim=-1)
                else:
                    predictions_sm = predictions
            top_labels = torch.argmax(predictions_sm, dim=1).tolist()
            return compute_gradmethod_preds_for_all_kNN(
                tst_feat = tst_chunk_dist,
                tst_chunk = tst_chunk,
                predictions = predictions,
                predictions_baseline = predictions_baseline,
                analysis_data = df_feat_for_expl, 
                saliency_map = explanations_chunk, 
                predict_fn = predict_fn, 
                n_closest = n_points_in_ball, 
                tree = tree,
                top_labels = top_labels,
                proba_output=proba_output
            )
        
class SmoothGradHandler(IntegratedGradientsHandler):
    def set_explainer(self, **kwargs):
        model = kwargs.get("model")
        multiply_by_inputs = kwargs.get("multiply_by_inputs")
        self.explainer = NoiseTunnel(IntegratedGradients(model, multiply_by_inputs=multiply_by_inputs))
