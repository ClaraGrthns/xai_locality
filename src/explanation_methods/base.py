import numpy as np
import os
from torch.utils.data import DataLoader, Subset
import time 
import torch

class BaseExplanationMethodHandler:
    def __init__(self,args):
        # self.explainer = self.set_explainer(**kwargs)
        self.args = args
    
    def set_explainer(self, **kwargs):
        raise NotImplementedError
    
    def explain_instance(self, **kwargs):
        raise NotImplementedError

    def compute_accuracy(self):
        raise NotImplementedError
    
    def compute_explanations(self):
        raise NotImplementedError
    
    def get_experiment_setting(self):
        raise NotImplementedError
    
    def prepare_data_for_analysis(self, dataset, df_feat):
        args = self.args
        indices = np.random.permutation(len(dataset))
        tst_indices, analysis_indices = np.split(indices, [args.max_test_points])
        print("using the following indices for testing: ", tst_indices)
        tst_data = Subset(dataset, tst_indices)
        analysis_data = Subset(dataset, analysis_indices)
        print("Length of data set for analysis", len(analysis_data))
        print("Length of test set", len(tst_data))
        if df_feat is not None:
            tst_feat, analysis_feat = np.split(df_feat[indices], [args.max_test_points])
        else:
            tst_feat, analysis_feat = tst_data.features, analysis_feat.features

        data_loader_tst = DataLoader(tst_data, batch_size=args.chunk_size, shuffle=False)
        return tst_feat, analysis_feat, data_loader_tst, analysis_data
    

    def iterate_over_data(self,
                     tst_feat_for_expl, 
                     tst_feat_for_dist, 
                     df_feat_for_expl, 
                     explanations, 
                     n_points_in_ball,
                     max_radius,
                     predict_fn, 
                     tree,
                     results_path,
                     experiment_setting,
                     results):
        """
        Base method for iterating over data to compute and store metrics.
        """
        chunk_size = int(np.min((self.args.chunk_size, len(tst_feat_for_expl))))
        tst_feat_for_expl_loader = DataLoader(tst_feat_for_expl, batch_size=chunk_size, shuffle=False)
        for i, batch in enumerate(tst_feat_for_expl_loader):
            start = time.time()
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(tst_feat_for_dist))
            print(f"Processing chunk {i}/{(len(tst_feat_for_expl) + chunk_size - 1) // chunk_size}")
            
            explanations_chunk = explanations[chunk_start:chunk_end]
            
            chunk_result = self.process_chunk(
                batch, 
                tst_feat_for_dist[chunk_start:chunk_end],
                df_feat_for_expl, 
                explanations_chunk, 
                predict_fn, 
                n_points_in_ball, 
                tree,
                max_radius
            )
            with torch.no_grad():
                y_preds = predict_fn(batch)

            if self.args.sample_around_instance: 
                metrics = self._calculate_metrics_R(chunk_result, y_preds)
            else:
                metrics = self._calculate_metrics_kNN(chunk_result, n_points_in_ball, y_preds)
            
            self._update_results_dict(results, metrics, chunk_start, chunk_end)
            
            self._save_chunk_results(results_path, experiment_setting, results)
            
            print(f"Processed chunk {i}/{(len(tst_feat_for_expl) + chunk_size - 1) // chunk_size}")
            print(f"Finished processing chunk {i} in {time.time() - start:.2f} seconds")

        return results
    
    def _calculate_metrics_kNN(self, chunk_result, n_points_in_ball, y_preds):
        """
        Calculate various metrics from chunk results.
        
        Args:
            chunk_result: Tuple containing prediction results from process_chunk
            n_points_in_ball: Number of neighbors to consider
            
        Returns:
            Dictionary of calculated metrics
        """
        if self.args.regression:
            model_preds, local_preds, dist = chunk_result
        else:
            model_preds, model_binary_preds, model_probs, local_preds, local_binary_preds, local_probs, dist = chunk_result
        
        max_distances = np.zeros((dist.shape[0], n_points_in_ball))
        for idx in range(n_points_in_ball):
            max_distances[:, idx] = np.max(dist[:, :idx+1], axis=1)
        
        kNNs = (np.arange(1, n_points_in_ball + 1))
        mean_model_preds = np.cumsum(model_preds, axis=1)/kNNs
        
        mse = np.cumsum(np.square(model_preds - local_preds), axis=1) / kNNs
        mae = np.cumsum(np.abs(model_preds - local_preds), axis=1) / kNNs
        var_model_preds = np.cumsum(np.square(model_preds - mean_model_preds), axis=1) / kNNs
        r2 = 1 - (mse / var_model_preds)
        
        res_dict_regression = {
            "mse": mse.T,
            "mae": mae.T,
            "r2": r2.T,
            "variance_logit": var_model_preds.T,
            "radius": max_distances.T,
        }
        if self.args.regression:
            if type(y_preds) == torch.Tensor:
                y_preds = y_preds.cpu().numpy()
            if y_preds.ndim == 1:
                y_preds = y_preds[:, None]
            mse_constant_clf = np.cumsum(np.square(model_preds - y_preds), axis=1) / kNNs #(num test samples,  num closest points)-(num test samples,1)
            mae_constant_clf = np.cumsum(np.abs(model_preds - y_preds), axis=1) / kNNs #(num test samples,  num closest points)-(num test samples,1)
            return {**res_dict_regression,
                    "mse_constant_clf": mse_constant_clf.T,
                    "mae_constant_clf": mae_constant_clf.T,
            }
           
        if not self.args.regression:
            mean_model_preds_probs = np.cumsum(model_probs, axis=1) / kNNs
            acc = np.cumsum((model_binary_preds == local_binary_preds), axis=1) / kNNs
            precision = np.cumsum(model_binary_preds * local_binary_preds, axis=1) / (np.cumsum(local_binary_preds, axis=1) + 1e-10)
            recall = np.cumsum(model_binary_preds * local_binary_preds, axis=1) / (np.cumsum(model_binary_preds, axis=1) + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            mse_proba = np.cumsum(np.square(model_probs - local_probs), axis=1) / kNNs
            mae_proba = np.cumsum(np.abs(model_probs - local_probs), axis=1) / kNNs

            var_model_preds_probs = np.cumsum(np.square(model_probs - mean_model_preds_probs), axis=1) / kNNs
            r2_proba = 1 - (mse_proba / var_model_preds_probs)

            ratio_of_ones = np.cumsum(model_binary_preds, axis=1)/kNNs
            all_ones_local_binary_preds = np.cumsum(local_binary_preds, axis=1) / kNNs
            gini_impurity = 1 - (np.square(ratio_of_ones)+ np.square(1 - ratio_of_ones))

            return {**res_dict_regression,
                    "accuraccy_constant_clf": ratio_of_ones.T,
                    "accuracy": acc.T,
                    "precision": precision.T,
                    "recall": recall.T,
                    "f1": f1.T,
                    "mse_proba": mse_proba.T,
                    "mae_proba": mae_proba.T,
                    "r2_proba": r2_proba.T,
                    "variance_proba": var_model_preds_probs.T,
                    "ratio_all_ones": ratio_of_ones.T,
                    "ratio_all_ones_local": all_ones_local_binary_preds.T,
                    "gini": gini_impurity.T
            }
        
    def _calculate_metrics_R(self, chunk_result, y_preds):
        """
        Calculate various metrics from chunk results.
        
        Args:
            chunk_result: Tuple containing prediction results from process_chunk
            n_points_in_ball: Number of neighbors to consider
            
        Returns:
            Dictionary of calculated metrics
        """
        if self.args.regression:
            model_preds, local_preds, radii = chunk_result
        else:
            model_preds, model_binary_preds, model_probs, local_preds, local_binary_preds, local_probs, radii = chunk_result
        
        
        mean_model_preds = np.nanmean(model_preds, axis=-1)
        
        mse = np.nanmean(np.square(model_preds - local_preds), axis=-1)
        mae = np.nanmean(np.abs(model_preds - local_preds), axis=-1)
        var_model_preds = np.nanvar(model_preds, axis=-1)
        r2 = 1 - (mse / var_model_preds)

        res_dict_regression = {
            "mse": mse.T,
            "mae": mae.T,
            "r2": r2.T,
            "variance_logit": var_model_preds.T,
            "radius": radii.T,
        }
        if self.args.regression:
            if type(y_preds) == torch.Tensor:
                y_preds = y_preds.cpu().numpy()
            if y_preds.ndim == 1:
                y_preds = y_preds[:, None]
            mse_constant_clf = np.nanmean(np.square(model_preds - y_preds[:, None, :]), axis=-1) #(num test samples,  num closest points)-(num test samples,1)
            mae_constant_clf = np.nanmean(np.abs(model_preds - y_preds[:, None, :]), axis=-1) #(num test samples,  num closest points)-(num test samples,1)
            return {**res_dict_regression,
                    "mse_constant_clf": mse_constant_clf.T,
                    "mae_constant_clf": mae_constant_clf.T,
            }
           
        if not self.args.regression:
            acc = np.nanmean((model_binary_preds == local_binary_preds), axis=-1)
            precision = np.nanmean(model_binary_preds * local_binary_preds, axis=-1) / (np.nanmean(local_binary_preds, axis=-1) + 1e-10)
            recall = np.nanmean(model_binary_preds * local_binary_preds, axis=-1) / (np.nanmean(model_binary_preds, axis=-1) + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            mse_proba = np.nanmean(np.square(model_probs - local_probs), axis=-1)
            mae_proba = np.nanmean(np.abs(model_probs - local_probs), axis=-1)

            var_model_preds_probs = np.nanvar(model_probs, axis=-1)
            r2_proba = 1 - (mse_proba / var_model_preds_probs)

            ratio_of_ones = np.nanmean(model_binary_preds, axis=-1)
            all_ones_local_binary_preds = np.nanmean(local_binary_preds, axis=-1)
            gini_impurity = 1 - (np.square(ratio_of_ones)+ np.square(1 - ratio_of_ones))

            return {**res_dict_regression,
                    "accuraccy_constant_clf": ratio_of_ones.T,
                    "accuracy": acc.T,
                    "precision": precision.T,
                    "recall": recall.T,
                    "f1": f1.T,
                    "mse_proba": mse_proba.T,
                    "mae_proba": mae_proba.T,
                    "r2_proba": r2_proba.T,
                    "variance_proba": var_model_preds_probs.T,
                    "ratio_all_ones": ratio_of_ones.T,
                    "ratio_all_ones_local": all_ones_local_binary_preds.T,
                    "gini": gini_impurity.T
            }
        
    
    def _update_results_dict(self, results, metrics, chunk_start, chunk_end):
        """
        Update the results dictionary with calculated metrics.
        
        Args:
            results: Dictionary to update
            metrics: Dictionary of calculated metrics
            chunk_start: Starting index of current chunk
            chunk_end: Ending index of current chunk
        """
        for key, value in metrics.items():
            if key in results:
                results[key][:, chunk_start:chunk_end] = value
    
    def _save_chunk_results(self, results_path, experiment_setting, results):
        """
        Save results to disk.
        
        Args:
            results_path: Path to save results
            experiment_setting: Experiment setting identifier
            results: Results dictionary to save
        """
        np.savez(os.path.join(results_path, experiment_setting), **results)
    
    def process_chunk(self, batch, tst_chunk_dist, df_feat_for_expl, explanations_chunk, predict_fn, n_points_in_ball, tree, chunk_start, chunk_end):
        """
        Process a single chunk of data. To be implemented by subclasses.
        
        Returns:
            Tuple containing (model_preds, model_binary_preds, model_probs, local_preds, local_binary_preds, local_probs, dist)
        """
        raise NotImplementedError("Subclasses must implement process_chunk")
    
    
    def update_results(self, results, idx, chunk_start, chunk_end, metrics):
        """Update the results dictionary with the computed metrics."""
        for key, value in metrics.items():
            if key in results:
                results[key][idx, chunk_start:chunk_end] = value
        return results
    
    def set_experiment_setting(self, max_fraction, max_radius=None):
        self.experiment_setting = self.get_experiment_setting(max_fraction, max_radius)
    
    def run_analysis(self, 
                     tst_feat_for_expl, 
                     tst_feat_for_dist, 
                     df_feat_for_expl, 
                     explanations, 
                     n_points_in_ball, 
                     max_radius,
                     predict_fn, 
                     tree,
                     results_path,
                     ):
        if self.args.sample_around_instance:
            range_n_points_in_ball = np.arange(n_points_in_ball)
        else: 
            range_n_points_in_ball = np.linspace(0.001, max_radius, n_points_in_ball) 
        experiment_setting = self.experiment_setting
        results_regression = {
            "mse": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "mae": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "r2": np.full((n_points_in_ball, self.args.max_test_points), np.nan),            
            "variance_logit": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "radius": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "n_points_in_ball": range_n_points_in_ball,
        }
        results_classification = {
            "accuraccy_constant_clf": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "accuracy": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "aucroc": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "precision": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "recall": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "f1": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            
            "mse_proba": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "mae_proba": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "r2_proba": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            
            "gini": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "variance_proba": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "ratio_all_ones": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            "ratio_all_ones_local": np.full((n_points_in_ball, self.args.max_test_points), np.nan),

        }

        if self.args.regression:
            results = {**results_regression, 
                       "mse_constant_clf": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
                       "mae_constant_clf": np.full((n_points_in_ball, self.args.max_test_points), np.nan),
            }
        else:
            results = {**results_classification, **results_regression}
            
        results = self.iterate_over_data(tst_feat_for_expl=tst_feat_for_expl, 
                 tst_feat_for_dist=tst_feat_for_dist, 
                 df_feat_for_expl=df_feat_for_expl, 
                 explanations=explanations, 
                 n_points_in_ball=n_points_in_ball, 
                 max_radius=max_radius,
                 predict_fn=predict_fn, 
                 tree=tree,
                 results_path=results_path,
                 experiment_setting=experiment_setting,
                 results=results)
        return results