import numpy as np
import torch
class BaseExplanationMethodHandler:
    def __init__(self, **kwargs):
        self.explainer = self.get_explainer(**kwargs)

    def get_explainer(self, **kwargs):
        raise NotImplementedError
    
    def explain_instance(self, **kwargs):
        raise NotImplementedError

    def compute_accuracy(self, tst_set, analysis_dataset, explanations, top_labels, explainer, predict_fn, n_closest, tree, pred_threshold):
        raise NotImplementedError
    
    def compute_explanations(self, results_path,  explainer, predict_fn, tst_data, device, args):
        raise NotImplementedError
    
    def get_experiment_setting(self, fractions, args):
        raise NotImplementedError
    
    def process_data(self):
        raise NotImplementedError