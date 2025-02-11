from src.explanation_methods.base import BaseExplanationMethodHandler
from captum.attr import IntegratedGradients, NoiseTunnel
import torch
from src.explanation_methods.gradient_methods.local_classifier import compute_gradmethod_mse_per_fraction, compute_saliency_maps
import os.path as osp
import os
import h5py
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset

class IntegratedGradientsHandler(BaseExplanationMethodHandler):
    def set_explainer(self, **kwargs):
        model = kwargs.get("model")
        self.explainer = IntegratedGradients(model, multiply_by_inputs=False)

    def explain_instance(self, **kwargs):
        return self.explainer.attribute(kwargs["input"], target=kwargs["target"])
    
    
    def compute_explanations(self, results_path, predict_fn, tst_data):
        device = torch.device("cpu")
        saliency_map_folder = osp.join(results_path, 
                                        "saliency_maps")
        saliency_map_file_path = osp.join(saliency_map_folder, f"saliency_map_{self.args.gradient_method}.h5")
        if osp.exists(saliency_map_file_path):
            with h5py.File(saliency_map_file_path, "r") as f:
                saliency_maps = f["saliency_map"][:]
            saliency_maps = torch.tensor(saliency_maps).float().to(device)
        else:
            if not osp.exists(saliency_map_folder):
                os.makedirs(saliency_map_folder)
            saliency_maps = compute_saliency_maps(self.explainer, predict_fn, tst_data, device)
            with h5py.File(saliency_map_file_path, "w") as f:
                f.create_dataset("saliency_map", data=saliency_maps.cpu().numpy())
        return saliency_maps
    
    def get_experiment_setting(self, fractions):
        return f"fractions-{np.round(fractions[0], 2)}-{np.round(fractions[-1], 2)}_grad_method-{self.args.gradient_method}_model_type-{self.args.model_type}_accuracy_fraction.npy"      
    
    def prepare_data_for_analysis(self, dataset, df_feat):
        args = self.args
        indices = np.random.permutation(len(dataset))
        tst_indices, analysis_indices = np.split(indices, [args.max_test_points])
        print("using the following indices for testing: ", tst_indices)
        tst_data = Subset(dataset, tst_indices)
        analysis_data = Subset(dataset, analysis_indices)
        tst_feat, analysis_feat = np.split(df_feat[indices], [args.max_test_points])
        data_loader_tst = DataLoader(tst_data, batch_size=args.chunk_size, shuffle=False)
        return tst_feat, analysis_feat, data_loader_tst, analysis_data

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
        
        
        chunk_size = self.args.chunk_size
        for i, (imgs,_,_) in enumerate(tst_feat_for_expl):
            chunk_start = i*chunk_size
            chunk_end = min(chunk_start + chunk_size, len(tst_feat_for_dist))
            print(f"Computing mse/var for chunk {i} from {chunk_start} to {chunk_end}")
            explanations_chunk = explanations[chunk_start:chunk_end]
            for n_closest in n_points_in_ball:
                print(f"Computing mse/var for {n_closest} closest points")
                # top_labels = torch.argmax(predict_fn(imgs), dim=1).tolist()
                with torch.no_grad():
                    predictions = predict_fn(imgs) #shape: num test samples x num classes
                    predictions_baseline = predict_fn(torch.zeros_like(imgs)) #
                chunk_result = compute_gradmethod_mse_per_fraction(tst_feat_for_dist[chunk_start:chunk_end], 
                                                                                    predictions,
                                                                                    predictions_baseline,
                                                                                    df_feat_for_expl, 
                                                                                    explanations_chunk, 
                                                                                    predict_fn, 
                                                                                    n_closest, 
                                                                                    tree, 
                                                                                )
                n_closest, mse, rad = chunk_result
                fraction_idx = np.where(n_points_in_ball == n_closest)[0][0]
                results["mse"][fraction_idx, i:i+chunk_size] = mse.cpu().numpy()
                results["radius"][fraction_idx, i:i+chunk_size] = rad
                print(f"Finished computing mse/var for {n_closest} closest points")
            np.savez(osp.join(results_path, experiment_setting), **results)


        
class SmoothGradHandler(IntegratedGradientsHandler):
    def set_explainer(self, **kwargs):
        model = kwargs.get("model")
        multiply_by_inputs = kwargs.get("multiply_by_inputs")
        self.explainer = NoiseTunnel(IntegratedGradients(model, multiply_by_inputs=multiply_by_inputs))
