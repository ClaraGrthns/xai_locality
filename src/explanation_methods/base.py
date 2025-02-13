import numpy as np
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
    
    def prepare_data_for_analysis(self):
        raise NotImplementedError
    
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
        
        fractions = n_points_in_ball/len(df_feat_for_expl)        
        experiment_setting = self.get_experiment_setting(fractions)
        num_fractions = len(n_points_in_ball)
        
        results = {
        "accuracy": np.zeros((num_fractions, self.args.max_test_points)),
        "mse": np.zeros((num_fractions, self.args.max_test_points)),
        "mse_lin_approx": np.zeros((num_fractions, self.args.max_test_points)),
        "variance_pred": np.zeros((num_fractions, self.args.max_test_points)),
        "radius": np.zeros((num_fractions, self.args.max_test_points)),
        "fraction_points_in_ball": fractions,
        "ratio_all_ones": np.zeros((num_fractions, self.args.max_test_points)),
        }  

        self.iterate_over_data(tst_feat_for_expl = tst_feat_for_expl, 
                     tst_feat_for_dist = tst_feat_for_dist, 
                     df_feat_for_expl = df_feat_for_expl, 
                     explanations = explanations, 
                     n_points_in_ball = n_points_in_ball, 
                     predict_fn = predict_fn, 
                     tree = tree,
                     results_path = results_path,
                     experiment_setting = experiment_setting,
                     results = results)