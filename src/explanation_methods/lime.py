from src.explanation_methods.base import BaseExplanationMethodHandler
import lime.lime_tabular
from src.explanation_methods.lime_analysis.lime_local_classifier import compute_lime_fidelity_per_kNN, compute_explanations
from src.utils.misc import get_path
import os.path as osp
import os
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import Subset, DataLoader
import time
import torch

class LimeHandler(BaseExplanationMethodHandler):
    def set_explainer(self, **kwargs):
        args = self.args
        trn_feat = kwargs.get('dataset')
        if type(trn_feat)== torch.Tensor:
            trn_feat = trn_feat.numpy()
        class_names = kwargs.get('class_names')
        self.explainer = lime.lime_tabular.LimeTabularExplainer(trn_feat, 
                                                    feature_names=np.arange(trn_feat.shape[1]),
                                                      class_names=class_names, 
                                                      discretize_continuous=True, 
                                                      random_state=args.random_seed, 
                                                      kernel_width=args.kernel_width)
    
    def explain_instance(self, **kwargs):
        return self.explainer.explain_instance(**kwargs)

    def compute_accuracy(self, tst_set, analysis_dataset, explanations, explainer, predict_fn, n_closest, tree, pred_threshold, top_labels=None):
        return compute_lime_fidelity_per_kNN(tst_set, analysis_dataset, explanations, explainer, predict_fn, n_closest, tree, pred_threshold)
    
    def compute_explanations(self, results_path, predict_fn, tst_data):
        args = self.args
        # Construct the explanation file name and path
        explanation_file_name = f"normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}"
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
            explanations = compute_explanations(self.explainer, tst_data, predict_fn, args.num_lime_features)
            
            # Save the explanations to the appropriate file
            np.save(explanation_file_path, explanations)
            print(f"Finished computing and saving explanations to: {explanation_file_path}")
        return explanations
    
    def get_experiment_setting(self, fractions):
        args = self.args
        df_setting = "complete_df" if args.include_trn and args.include_val else "only_test"
        experiment_setting = f"fractions-{0}-{np.round(fractions[-1])}_{df_setting}_kernel_width-{args.kernel_width}_model_regr-{args.model_regressor}_model_type-{args.model_type}_accuracy_fraction.npy"
        # if args.num_lime_features > 10:
        #     experiment_setting = f"num_features-{args.num_lime_features}_{experiment_setting}"
        # if args.num_test_splits > 1:
        #     experiment_setting = f"split-{args.split_idx}_{experiment_setting}"
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
        print(len(tst_feat_for_expl))
        print(len(df_feat_for_expl))
        chunk_size = self.args.chunk_size
        predict_threshold = self.args.predict_threshold
        tst_feat_for_expl_loader = DataLoader(tst_feat_for_expl, batch_size=chunk_size, shuffle=False)
        for i, batch in enumerate(tst_feat_for_expl_loader):
            tst_chunk = batch[0].numpy()
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(df_feat_for_expl))
            print(f"Processing chunk {i // chunk_size + 1}/{(len(tst_feat_for_expl) + chunk_size - 1) // chunk_size}")
            explanations_chunk = explanations[chunk_start:chunk_end]

            if self.args.debug:
                # Normal for loop for easier debugging
                for n_closest in n_points_in_ball:
                    start_time = time.time()
                    print(f"Processing {n_closest} nearest neighbours")
                    n_closest, res_binary_classification, res_regression, res_impurity, R = compute_lime_fidelity_per_kNN(
                    tst_chunk, df_feat_for_expl, explanations_chunk, self.explainer, predict_fn, n_closest, tree, predict_threshold
                    )
                    aucroc, acc, precision, recall, f1 = res_binary_classification
                    mse, mae, r2 = res_regression
                    gini, ratio, variance = res_impurity
                    fraction_idx = np.where(n_points_in_ball == n_closest)[0][0]
                    
                    results["aucroc"][fraction_idx, chunk_start:chunk_end] = aucroc
                    results["accuracy"][fraction_idx, chunk_start:chunk_end] = acc
                    results["precision"][fraction_idx, chunk_start:chunk_end] = precision
                    results["recall"][fraction_idx, chunk_start:chunk_end] = recall
                    results["f1"][fraction_idx, chunk_start:chunk_end] = f1
                    
                    results["mse"][fraction_idx, chunk_start:chunk_end] = mse
                    results["mae"][fraction_idx, chunk_start:chunk_end] = mae
                    results["r2"][fraction_idx, chunk_start:chunk_end] = r2

                    results["gini"][fraction_idx, chunk_start:chunk_end] = gini
                    results["ratio_all_ones"][fraction_idx, chunk_start:chunk_end] = ratio
                    results["variance"][fraction_idx, chunk_start:chunk_end] = variance

                    results["radius"][fraction_idx, chunk_start:chunk_end] = R

                    print(f"Time taken: {time.time() - start_time}")
            else:
                chunk_results = Parallel(n_jobs=-1)(
                    delayed(compute_lime_fidelity_per_kNN)(
                    tst_chunk, df_feat_for_expl, explanations_chunk, self.explainer, predict_fn, n_closest, tree, predict_threshold
                    )
                    for n_closest in n_points_in_ball
                )
                # Unpack results directly into the correct positions in the arrays
                for n_closest, res_binary_classification, res_regression, res_impurity, R  in chunk_results:
                    aucroc, acc, precision, recall, f1 = res_binary_classification
                    mse, mae, r2 = res_regression
                    gini, ratio, variance = res_impurity

                    fraction_idx = np.where(n_points_in_ball == n_closest)[0][0]
                    results["aucroc"][fraction_idx, chunk_start:chunk_end] = aucroc
                    results["accuracy"][fraction_idx, chunk_start:chunk_end] = acc
                    results["precision"][fraction_idx, chunk_start:chunk_end] = precision
                    results["recall"][fraction_idx, chunk_start:chunk_end] = recall
                    results["f1"][fraction_idx, chunk_start:chunk_end] = f1
                    
                    results["mse"][fraction_idx, chunk_start:chunk_end] = mse
                    results["mae"][fraction_idx, chunk_start:chunk_end] = mae
                    results["r2"][fraction_idx, chunk_start:chunk_end] = r2

                    results["gini"][fraction_idx, chunk_start:chunk_end] = gini
                    results["ratio_all_ones"][fraction_idx, chunk_start:chunk_end] = ratio
                    results["variance"][fraction_idx, chunk_start:chunk_end] = variance

                    results["radius"][fraction_idx, chunk_start:chunk_end] = R

            np.savez(osp.join(results_path, experiment_setting), **results)
            print(f"Processed chunk {i // chunk_size + 1}/{(len(tst_feat_for_expl) + chunk_size - 1) // chunk_size}")
                
        return results
            

