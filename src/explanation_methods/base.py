import numpy as np
import os
from torch.utils.data import DataLoader, Subset

from src.utils.metrics import regression_metrics_per_row
from src.utils.metrics import impurity_metrics_per_row
from src.utils.metrics import binary_classification_metrics_per_row

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
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(tst_feat_for_dist))
            print(f"Processing chunk {i}/{(len(tst_feat_for_expl) + chunk_size - 1) // chunk_size}")
            
            explanations_chunk = explanations[chunk_start:chunk_end]
            
            # Process the chunk according to the specific model requirements
            chunk_result = self.process_chunk(
                batch, 
                tst_feat_for_dist[chunk_start:chunk_end],
                df_feat_for_expl, 
                explanations_chunk, 
                predict_fn, 
                n_points_in_ball, 
                tree,
                chunk_start,
                chunk_end
            )
            
            # Unpack results based on the format returned by the specific implementation
            model_preds, model_binary_preds, model_probs, local_preds, local_binary_preds, local_probs, dist = chunk_result
            

            # Precompute max distances for each cumulative neighborhood size
            # For each sample, calculate the maximum distance across progressively more neighbors
            max_distances = np.zeros((dist.shape[0], n_points_in_ball))
            for idx in range(n_points_in_ball):
                max_distances[:, idx] = np.max(dist[:, :idx+1], axis=1)
            
            # Compute metrics for each neighborhood size
            for idx in range(n_points_in_ball):
                # Get current neighborhood slice
                current_slice = slice(0, idx + 1)
                
                # Binary classification metrics
                aucroc, acc, precision, recall, f1 = self.binary_classification_metrics(
                    model_binary_preds[:, current_slice],
                    local_binary_preds[:, current_slice],
                    local_probs[:, current_slice]
                )
                
                # Regression metrics
                mse, mae, r2 = self.regression_metrics(
                    model_preds[:, current_slice],
                    local_preds[:, current_slice]
                )
                
                # Probability regression metrics
                mse_proba, mae_proba, r2_proba = self.regression_metrics(
                    model_probs[:, current_slice],
                    local_probs[:, current_slice]
                )
                
                # Impurity metrics
                gini, ratio = self.impurity_metrics(model_binary_preds[:, current_slice])
                
                # Variance metrics
                variance_preds = np.var(model_preds[:, current_slice], axis=1)
                variance_pred_proba = np.var(model_probs[:, current_slice], axis=1)
                
                # Constant classifier accuracy
                acc_constant_clf = np.mean(model_binary_preds[:, current_slice], axis=1)
                all_ones_local_binary_preds = np.mean(local_binary_preds[:, current_slice], axis=1)
                
                # Store results for current neighborhood size
                self.update_results(results, idx, chunk_start, chunk_end, {
                    "accuraccy_constant_clf": acc_constant_clf,
                    "aucroc": aucroc,
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "mse_proba": mse_proba,
                    "mae_proba": mae_proba,
                    "r2_proba": r2_proba,
                    "gini": gini,
                    "ratio_all_ones": ratio,
                    "ratio_all_ones_local": all_ones_local_binary_preds,
                    "variance_proba": variance_pred_proba,
                    "variance_logit": variance_preds,
                    "radius": max_distances[:, idx],  # Use precomputed max distances
                })
            
            # Save results after processing each chunk
            np.savez(os.path.join(results_path, experiment_setting), **results)
            print(f"Processed chunk {i}/{(len(tst_feat_for_expl) + chunk_size - 1) // chunk_size}")
            
        return results
    
    def process_chunk(self, batch, tst_chunk_dist, df_feat_for_expl, explanations_chunk, predict_fn, n_points_in_ball, tree, chunk_start, chunk_end):
        """
        Process a single chunk of data. To be implemented by subclasses.
        
        Returns:
            Tuple containing (model_preds, model_binary_preds, model_probs, local_preds, local_binary_preds, local_probs, dist)
        """
        raise NotImplementedError("Subclasses must implement process_chunk")
    
    def binary_classification_metrics(self, model_binary_preds, local_binary_preds, local_probs):
        """Calculate binary classification metrics."""
        return binary_classification_metrics_per_row(model_binary_preds, local_binary_preds, local_probs)
    
    def regression_metrics(self, model_preds, local_preds):
        """Calculate regression metrics."""
        return regression_metrics_per_row(model_preds, local_preds)
    
    def impurity_metrics(self, model_binary_preds):
        """Calculate impurity metrics."""
        return impurity_metrics_per_row(model_binary_preds)
    
    def update_results(self, results, idx, chunk_start, chunk_end, metrics):
        """Update the results dictionary with the computed metrics."""
        for key, value in metrics.items():
            if key in results:
                results[key][idx, chunk_start:chunk_end] = value
        return results
    
    def set_experiment_setting(self, max_fraction):
        self.experiment_setting = self.get_experiment_setting(max_fraction)
    
    def run_analysis(self, 
                     tst_feat_for_expl, 
                     tst_feat_for_dist, 
                     df_feat_for_expl, 
                     explanations, 
                     n_points_in_ball, 
                     predict_fn, 
                     tree,
                     results_path,
                     ):
        
        max_fraction = n_points_in_ball/len(df_feat_for_expl)        
        experiment_setting = self.get_experiment_setting(max_fraction)
        

        num_fractions = len(np.arange(n_points_in_ball))
        results = {
            "accuraccy_constant_clf": np.zeros((num_fractions, self.args.max_test_points)),
            "accuracy": np.zeros((num_fractions, self.args.max_test_points)),
            "aucroc": np.zeros((num_fractions, self.args.max_test_points)),
            "precision": np.zeros((num_fractions, self.args.max_test_points)),
            "recall": np.zeros((num_fractions, self.args.max_test_points)),
            "f1": np.zeros((num_fractions, self.args.max_test_points)),
            
            "mse": np.zeros((num_fractions, self.args.max_test_points)),
            "mae": np.zeros((num_fractions, self.args.max_test_points)),
            "r2": np.zeros((num_fractions, self.args.max_test_points)),

            "mse_proba": np.zeros((num_fractions, self.args.max_test_points)),
            "mae_proba": np.zeros((num_fractions, self.args.max_test_points)),
            "r2_proba": np.zeros((num_fractions, self.args.max_test_points)),
            
            "gini": np.zeros((num_fractions, self.args.max_test_points)),
            "variance_proba": np.zeros((num_fractions, self.args.max_test_points)),
            "variance_logit": np.zeros((num_fractions, self.args.max_test_points)),
            "ratio_all_ones": np.zeros((num_fractions, self.args.max_test_points)),
            "ratio_all_ones_local": np.zeros((num_fractions, self.args.max_test_points)),
            
            "radius": np.zeros((num_fractions, self.args.max_test_points)),
            "n_points_in_ball": np.arange(n_points_in_ball),
        }
        results = self.iterate_over_data(tst_feat_for_expl = tst_feat_for_expl, 
                     tst_feat_for_dist = tst_feat_for_dist, 
                     df_feat_for_expl = df_feat_for_expl, 
                     explanations = explanations, 
                     n_points_in_ball = n_points_in_ball, 
                     predict_fn = predict_fn, 
                     tree = tree,
                     results_path = results_path,
                     experiment_setting = experiment_setting,
                     results = results)
        return results