from src.explanation_methods.base import BaseExplanationMethodHandler
from captum.attr import IntegratedGradients, NoiseTunnel
import torch
from src.explanation_methods.gradient_methods.local_classifier import compute_gradmethod_accuracy_per_fraction, compute_saliency_maps
import os.path as osp
import os
import h5py
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset

class IntegratedGradientsHandler(BaseExplanationMethodHandler):
    def get_explainer(self, **kwargs):
        model = kwargs.get("model")
        multiply_by_inputs = kwargs.get("multiply_by_inputs")
        return IntegratedGradients(model, multiply_by_inputs=multiply_by_inputs)

    def explain_instance(self, **kwargs):
        return self.explainer.attribute(kwargs["input"], target=kwargs["target"])
    
    def compute_accuracy(self, tst_set, analysis_dataset, explanations, top_labels, predict_fn, n_closest, tree, pred_threshold, explainer=None):
        return compute_gradmethod_accuracy_per_fraction(tst_set, top_labels, analysis_dataset, explanations, predict_fn, n_closest, tree, device="cpu", pred_threshold=pred_threshold)
    
    def compute_explanations(self, results_path, explainer, predict_fn, tst_data, device, args):
        saliency_map_folder = osp.join(results_path, 
                                        "saliency_maps")
        saliency_map_file_path = osp.join(saliency_map_folder, f"saliency_map_{args.gradient_method}.h5")
        if osp.exists(saliency_map_file_path):
            with h5py.File(saliency_map_file_path, "r") as f:
                saliency_maps = f["saliency_map"][:]
            saliency_maps = torch.tensor(saliency_maps).float().to(device)
        else:
            if not osp.exists(saliency_map_folder):
                os.makedirs(saliency_map_folder)
            saliency_maps = compute_saliency_maps(explainer, predict_fn, tst_data, device)
            with h5py.File(saliency_map_file_path, "w") as f:
                f.create_dataset("saliency_map", data=saliency_maps.cpu().numpy())
        return saliency_maps
    
    def get_experiment_setting(self, fractions, args):
        return f"fractions-{0}-{np.round(fractions[-1])}_grad_method-{args.gradient_method}_model_type-{args.model_type}_accuracy_fraction.npy"      
    
    def process_data(self, dataset, model_handler, args):
        df_feat = model_handler.load_feature_vectors()
        indices = np.random.permutation(len(dataset))
        tst_indices, analysis_indices = np.split(indices, [args.max_test_points])
        tst_data = Subset(dataset, tst_indices)
        analysis_data = Subset(dataset, analysis_indices)
        tst_feat, analysis_feat = np.split(df_feat[indices], [args.max_test_points])
        data_loader_tst = DataLoader(tst_data, batch_size=args.chunk_size, shuffle=False)
        return tst_feat, analysis_feat, data_loader_tst, analysis_data, 


        
class SmoothGradHandler(IntegratedGradientsHandler):
    def get_explainer(self, **kwargs):
        model = kwargs.get("model")
        multiply_by_inputs = kwargs.get("multiply_by_inputs")
        return NoiseTunnel(IntegratedGradients(model, multiply_by_inputs=multiply_by_inputs))
