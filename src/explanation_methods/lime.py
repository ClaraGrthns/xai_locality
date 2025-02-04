from src.explanation_methods.base import BaseExplanationMethodHandler
import lime.lime_tabular
from src.explanation_methods.lime_analysis.lime_local_classifier import compute_lime_accuracy_per_fraction, compute_explanations
import os.path as osp
import os
import numpy as np

class LimeHandler(BaseExplanationMethodHandler):
    def get_explainer(self, **kwargs):
        trn_feat = kwargs.get('trn_feat')
        feature_names = kwargs.get('feature_names')
        class_names = kwargs.get('class_names')
        discretize_continuous = kwargs.get('discretize_continuous')
        random_state = kwargs.get('random_state')
        kernel_width = kwargs.get('kernel_width')
        return lime.lime_tabular.LimeTabularExplainer(trn_feat, 
                                                      feature_names=feature_names, 
                                                      class_names=class_names, 
                                                      discretize_continuous=discretize_continuous, 
                                                      random_state=random_state, 
                                                      kernel_width=kernel_width)

    def explain_instance(self, **kwargs):
        return self.explainer.explain_instance(**kwargs)

    def compute_accuracy(self, tst_set, analysis_dataset, explanations, explainer, predict_fn, n_closest, tree, pred_threshold, top_labels=None):
        return compute_lime_accuracy_per_fraction(tst_set, analysis_dataset, explanations, explainer, predict_fn, n_closest, tree, pred_threshold)
    
    def compute_explanations(self, results_path,  explainer, predict_fn, tst_data,args, device=None):
        # Construct the explanation file name and path
        explanation_file_name = f"normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}"
        if args.num_lime_features > 10:
            explanation_file_name += f"_num_features-{args.num_lime_features}"
        if args.num_test_splits > 1:
            explanation_file_name = f"split-{args.split_idx}_{explanation_file_name}"
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
            print("Precomputed explanations not found. Computing explanations for the test set...")
            explanations = compute_explanations(explainer, tst_data, predict_fn, args.num_lime_features)
            
            # Save the explanations to the appropriate file
            np.save(explanation_file_path, explanations)
            print(f"Finished computing and saving explanations to: {explanation_file_path}")
        return explanations
    
    def get_experiment_setting(self, fractions, args):
        df_setting = "complete_df" if args.include_trn and args.include_val else "only_test"

        experiment_setting = f"fractions-{0}-{np.round(fractions[-1])}_{df_setting}_kernel_width-{args.kernel_width}_model_regr-{args.model_regressor}_model_type-{args.model_type}_accuracy_fraction.npy"
        if args.num_lime_features > 10:
            experiment_setting = f"num_features-{args.num_lime_features}_{experiment_setting}"
        if args.num_test_splits > 1:
            experiment_setting = f"split-{args.split_idx}_{experiment_setting}"
        return experiment_setting
    
    def process_data(self, dataset, model_handler, args):
        tst_feat, _, val_feat, _, trn_feat, _  = dataset
        df_feat = tst_feat[args.max_test_points:]
        tst_feat = tst_feat[: args.max_test_points]
        if args.include_trn:
            df_feat = np.concatenate([trn_feat, df_feat], axis=0)
        if args.include_val:
            df_feat = np.concatenate([df_feat, val_feat], axis=0)
        return tst_feat, df_feat, tst_feat, df_feat


            

