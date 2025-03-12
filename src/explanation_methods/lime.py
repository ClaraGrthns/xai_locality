from src.explanation_methods.base import BaseExplanationMethodHandler
import lime.lime_tabular
from src.explanation_methods.lime_analysis.lime_local_classifier import compute_lime_fidelity_per_kNN, compute_explanations, get_lime_preds_for_all_kNN
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
        self.explainer = lime.lime_tabular.LimeTabularExplainer(trn_feat, 
                                                    feature_names=np.arange(trn_feat.shape[1]),
                                                      class_names=class_names, 
                                                      discretize_continuous=True, 
                                                      random_state=args.random_seed, 
                                                      kernel_width=args.kernel_width)
    
    def explain_instance(self, **kwargs):
        return self.explainer.explain_instance(**kwargs)

    
    def compute_explanations(self, results_path, predict_fn, tst_data):
        args = self.args
        # Construct the explanation file name and path
        if args.distance_measure in ["euclidean", "minowski", "l2"]:
            explanation_file_name = f"normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}"
        else:
            explanation_file_name = f"normalized_data_explanations_test_set_kernel_width-{args.kernel_width}_model_regressor-{args.model_regressor}_distance_measure-{args.distance_measure}"

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
        chunk_size = int(np.min((self.args.chunk_size, len(tst_feat_for_expl))))
        predict_threshold = self.args.predict_threshold
        tst_feat_for_expl_loader = DataLoader(tst_feat_for_expl, batch_size=chunk_size, shuffle=False)
        for i, batch in enumerate(tst_feat_for_expl_loader):
            tst_chunk = batch.numpy()#[0]
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(df_feat_for_expl))
            print(f"Processing chunk {i}/{(len(tst_feat_for_expl) + chunk_size - 1) // chunk_size}")
            explanations_chunk = explanations[chunk_start:chunk_end]
            res = get_lime_preds_for_all_kNN(tst_chunk, 
                                             df_feat_for_expl, 
                                             explanations_chunk, 
                                             self.explainer, 
                                             predict_fn, 
                                             n_points_in_ball, 
                                             tree, 
                                             predict_threshold)
            model_predicted_top_label, model_prob_of_top_label, local_preds_label, local_preds, dist = res
            for idx in range(n_points_in_ball):
                # 4. Compute metrics for binary and regression tasks
                R = np.max(dist[:,:idx+1], axis=-1)
                

                aucroc, acc, precision, recall, f1 = binary_classification_metrics_per_row(model_predicted_top_label[:,:idx+1], 
                                                                                local_preds_label[:,:idx+1],
                                                                                local_preds[:,:idx+1]
                                                                                )
                mse, mae, r2 = regression_metrics_per_row(model_prob_of_top_label[:,:idx+1],
                                                            local_preds[:,:idx+1])
                gini, ratio = impurity_metrics_per_row(model_predicted_top_label[:,:idx+1])
                variance_preds = np.var(model_prob_of_top_label[:,:idx+1], axis=1)

                acc_constant_clf = np.mean(model_predicted_top_label[:, :idx+1], axis=1)

                results["accuraccy_constant_clf"][idx, chunk_start:chunk_end] = acc_constant_clf
                   
                results["aucroc"][idx, chunk_start:chunk_end] = aucroc
                results["accuracy"][idx, chunk_start:chunk_end] = acc
                results["precision"][idx, chunk_start:chunk_end] = precision
                results["recall"][idx, chunk_start:chunk_end] = recall
                results["f1"][idx, chunk_start:chunk_end] = f1
                
                results["mse_proba"][idx, chunk_start:chunk_end] = mse
                results["mae_proba"][idx, chunk_start:chunk_end] = mae
                results["r2_proba"][idx, chunk_start:chunk_end] = r2

                results["gini"][idx, chunk_start:chunk_end] = gini
                results["ratio_all_ones"][idx, chunk_start:chunk_end] = ratio
                results["variance_proba"][idx, chunk_start:chunk_end] = variance_preds

                results["radius"][idx, chunk_start:chunk_end] = R

           
            np.savez(osp.join(results_path, experiment_setting), **results)
            print(f"Processed chunk {i}/{(len(tst_feat_for_expl) + chunk_size - 1) // chunk_size}")
        return results
            

