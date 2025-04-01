import numpy as np
import os
from torch.utils.data import DataLoader, Subset

from src.utils.metrics import regression_metrics_per_row
from src.utils.metrics import impurity_metrics_per_row
from src.utils.metrics import binary_classification_metrics_per_row
import time 

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
                chunk_start,
                chunk_end
            )
            model_preds, model_binary_preds, model_probs, local_preds, local_binary_preds, local_probs, dist = chunk_result

            # For each sample, calculate the maximum distance across progressively more neighbors
            max_distances = np.zeros((dist.shape[0], n_points_in_ball))
            for idx in range(n_points_in_ball):
                max_distances[:, idx] = np.max(dist[:, :idx+1], axis=1)
            
            kNNs = (np.arange(1, n_points_in_ball + 1))
            mean_model_preds = np.cumsum(model_preds, axis=1)/kNNs
            mean_model_preds_probs = np.cumsum(model_probs, axis=1) / kNNs
            
            acc = np.cumsum((model_binary_preds == local_binary_preds), axis=1) / kNNs
            precision = np.cumsum(model_binary_preds * local_binary_preds, axis=1) / (np.cumsum(local_binary_preds, axis=1) + 1e-10)
            recall = np.cumsum(model_binary_preds * local_binary_preds, axis=1) / (np.cumsum(model_binary_preds, axis=1) + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

            mse = np.cumsum(np.square(model_preds - local_preds), axis=1) / kNNs
            mae = np.cumsum(np.abs(model_preds - local_preds), axis=1) / kNNs

            mse_proba = np.cumsum(np.square(model_probs - local_probs), axis=1) / kNNs
            mae_proba = np.cumsum(np.abs(model_probs - local_probs), axis=1) / kNNs

            var_model_preds = np.cumsum(np.square(model_preds - mean_model_preds), axis=1) / kNNs
            var_model_preds_probs = np.cumsum(np.square(model_probs - mean_model_preds_probs), axis=1) / kNNs
            r2 = 1 - (mse / var_model_preds)
            r2_proba = 1 - (mse_proba / var_model_preds_probs)

            ratio_of_ones = np.cumsum(model_binary_preds, axis=1)/kNNs
            all_ones_local_binary_preds = np.cumsum(local_binary_preds, axis=1) / kNNs

            gini_impurity = 1 - (np.square(ratio_of_ones)+ np.square(1 - ratio_of_ones))

            results["accuraccy_constant_clf"][:, chunk_start:chunk_end] = ratio_of_ones.T
            results["accuracy"][:,chunk_start:chunk_end] = acc.T
            results["precision"][:,chunk_start:chunk_end] = precision.T
            results["recall"][:,chunk_start:chunk_end] = recall.T
            results["f1"][:,chunk_start:chunk_end] = f1.T
            results["mse"][:,chunk_start:chunk_end] = mse.T
            results["mae"][:,chunk_start:chunk_end] = mae.T
            results["r2"][:,chunk_start:chunk_end] = r2.T
            results["mse_proba"][:,chunk_start:chunk_end] = mse_proba.T
            results["mae_proba"][:,chunk_start:chunk_end] = mae_proba.T
            results["r2_proba"][:,chunk_start:chunk_end] = r2_proba.T
            results["variance_proba"][:,chunk_start:chunk_end] = var_model_preds_probs.T
            results["variance_logit"][:,chunk_start:chunk_end] = var_model_preds.T
            results["radius"][:,chunk_start:chunk_end] = max_distances.T
            results["ratio_all_ones"][:,chunk_start:chunk_end] = ratio_of_ones.T
            results["ratio_all_ones_local"][:,chunk_start:chunk_end] = all_ones_local_binary_preds.T
            results["gini"][:,chunk_start:chunk_end] = gini_impurity.T

            # Save results after processing each chunk
            np.savez(os.path.join(results_path, experiment_setting), **results)
            print(f"Processed chunk {i}/{(len(tst_feat_for_expl) + chunk_size - 1) // chunk_size}")
            print(f"Finished processing chunk {i} in {time.time() - start:.2f} seconds")

            
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