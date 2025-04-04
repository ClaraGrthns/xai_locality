from src.explanation_methods.base import BaseExplanationMethodHandler
import lime.lime_tabular
from src.explanation_methods.lime_analysis.lime_local_classifier import compute_explanations, get_lime_preds_for_all_kNN, get_lime_rergression_preds_for_all_kNN
from src.utils.misc import get_path
import os.path as osp
import os
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import Subset, DataLoader
import time
import torch
from src.utils.metrics import binary_classification_metrics_per_row, regression_metrics_per_row, impurity_metrics_per_row

class LimeHandler(BaseExplanationMethodHandler):
    def set_explainer(self, **kwargs):
        args = self.args
        trn_feat = kwargs.get('dataset')
        if type(trn_feat)== torch.Tensor:
            trn_feat = trn_feat.numpy()
        class_names = kwargs.get('class_names')
        mode = "regression" if args.regression else "classification"
        self.explainer = lime.lime_tabular.LimeTabularExplainer(trn_feat, 
                                                    feature_names=np.arange(trn_feat.shape[1]),
                                                      class_names=class_names, 
                                                      discretize_continuous=True, 
                                                      mode=mode, 
                                                      random_state=args.random_seed, 
                                                      kernel_width=args.kernel_width)
    
    def explain_instance(self, **kwargs):
        return self.explainer.explain_instance(**kwargs)

    
    def compute_explanations(self, results_path, predict_fn, tst_data):
        args = self.args
        # Construct the explanation file name and path
        explanation_file_name = f"normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}_distance_measure-{args.distance_measure}"
        if args.num_lime_features > 10:
            explanation_file_name += f"_num_features-{args.num_lime_features}"
        # if args.num_lime_features > 10:
        #     explanation_file_name += f"_num_features-{args.num_lime_features}"
        # if args.num_test_splits > 1:
        #     explanation_file_name = f"split-{args.split_idx}_{explanation_file_name}"
        explanations_dir = osp.join(results_path, "explanations")
        explanation_file_path = osp.join(explanations_dir, explanation_file_name)
        print(f"using explanation path: {explanation_file_path}")

        if not osp.exists(explanations_dir):
            os.makedirs(explanations_dir)
        
        if osp.exists(explanation_file_path+".npy"):
            print(f"Using precomputed explanations from: {explanation_file_path}")
            explanations = np.load(explanation_file_path+".npy", allow_pickle=True)
            print(f"{len(explanations)} explanations loaded")
        else:
            tst_data = tst_data.features
            print("Precomputed explanations not found. Computing explanations for the test set...")
            explanations = compute_explanations(self.explainer, tst_data, predict_fn, args.num_lime_features, sequential_computation=args.debug, distance_metric=args.distance_measure)
            
            # Save the explanations to the appropriate file
            np.save(explanation_file_path, explanations)
            print(f"Finished computing and saving explanations to: {explanation_file_path}")
        return explanations
    
    
    def get_experiment_setting(self, fractions):
        args = self.args
        df_setting = "complete_df" if args.include_trn and args.include_val else "only_test"
        experiment_setting = f"fractions-{0}-{np.round(fractions, 2)}_{df_setting}_kernel_width-{args.kernel_width}_model_regr-{args.model_regressor}_model_type-{args.model_type}_dist_measure-{args.distance_measure}_accuracy_fraction"
        if self.args.num_lime_features > 10:
            experiment_setting += f"_num_features-{self.args.num_lime_features}"
        if self.args.regression:
            experiment_setting = "regression_" + experiment_setting
        return experiment_setting
    
    # def prepare_data_for_analysis(self, dataset, df_feat):
    #     args = self.args
    #     tst_feat, _, val_feat, _, trn_feat, _  = dataset
    #     df_feat = tst_feat[args.max_test_points:]
    #     tst_feat = tst_feat[: args.max_test_points]
    #     if args.include_trn:
    #         df_feat = np.concatenate([trn_feat, df_feat], axis=0)
    #     if args.include_val:
    #         df_feat = np.concatenate([df_feat, val_feat], axis=0)
    #     return tst_feat, df_feat, tst_feat, df_feat
    
    def process_chunk(self, batch, tst_chunk_dist, df_feat_for_expl, explanations_chunk, predict_fn, n_points_in_ball, tree):
        """
        Process a single chunk of data for LIME method.
        """
        tst_chunk = batch.numpy()  # For LIME method, convert batch to numpy
        predict_threshold = self.args.predict_threshold

        if self.args.regression:
            return get_lime_rergression_preds_for_all_kNN(
                tst_chunk, 
                df_feat_for_expl, 
                explanations_chunk, 
                self.explainer, 
                predict_fn, 
                n_points_in_ball, 
                tree, 
            )
        else:

            res = get_lime_preds_for_all_kNN(
                tst_chunk, 
                df_feat_for_expl, 
                explanations_chunk, 
                self.explainer, 
                predict_fn, 
                n_points_in_ball, 
                tree, 
                predict_threshold
            )
            
            model_predicted_top_label, model_prob_of_top_label, local_preds_label, local_preds, dist = res
            
            # Reformat to match the expected output format of process_chunk
            return (
                model_prob_of_top_label,  # model_preds 
                model_predicted_top_label,  # model_binary_preds
                model_prob_of_top_label,  # model_probs
                local_preds,  # local_preds
                local_preds_label,  # local_binary_preds
                local_preds,  # local_probs
                dist  # dist
            )



