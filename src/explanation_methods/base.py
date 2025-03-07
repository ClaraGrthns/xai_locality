import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import os
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
    

    def iterate_over_data(self):
        raise NotImplementedError
    
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
        experiment_path = os.path.join(results_path, experiment_setting +".npz")
        if os.path.exists(experiment_path):
            print(f"Experiment with setting {experiment_setting} already exists.")
            exit(-1)

        num_fractions = len(np.arange(n_points_in_ball))
        results = {
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