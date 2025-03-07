import os
import numpy as np

def get_results_files_dict(explanation_method: str, models: list[str], datasets: list[str]) -> dict:
    results_folder = f"/home/grotehans/xai_locality/results/{explanation_method}"
    results_files_dict = {}
    
    for model in models:
        results_files_dict[model] = {}
        for dataset in datasets:
            path_to_results = os.path.join(results_folder, model, dataset)
            if not os.path.exists(path_to_results):
                continue
            res = [os.path.join(path_to_results, f) for f in os.listdir(f"{results_folder}/{model}/{dataset}") if f.endswith('fraction.npz')]
            if len(res) > 0:
                results_files_dict[model][dataset] = res[0]
    
    # Rename synthetic dataset keys
    for model in results_files_dict:
        keys = list(results_files_dict[model].keys())
        for key in keys:
            if "synthetic_data/n_feat50_n_informative2" in key:
                results_files_dict[model]["synthetic (simple)"] = results_files_dict[model].pop(key)
            elif "synthetic_data/n_feat50_n_informative10" in key:
                results_files_dict[model]["synthetic (medium)"] = results_files_dict[model].pop(key)
            elif "synthetic_data/n_feat100" in key:
                results_files_dict[model]["synthetic (complex)"] = results_files_dict[model].pop(key)
                
    return results_files_dict
def get_non_zero_cols(array):
    return array.shape[1] - np.sum(np.all(array == 0, axis=0))


def load_and_get_non_zero_cols(data_path):
    results = np.load(data_path, allow_pickle=True)
    nr_non_zero_columns = get_non_zero_cols(results['accuracy']) 
    n_points_in_ball = results['n_points_in_ball']
    # Common metrics for both methods
    accuracy = results['accuracy'][:, :nr_non_zero_columns]
    precision = results['precision'][:, :nr_non_zero_columns]
    recall = results['recall'][:, :nr_non_zero_columns]
    f1 = results['f1'][:, :nr_non_zero_columns]
    mse_proba = results['mse_proba'][:, :nr_non_zero_columns]
    mae_proba = results['mae_proba'][:, :nr_non_zero_columns]
    r2_proba = results['r2_proba'][:, :nr_non_zero_columns]
    gini = results["gini"][:, :nr_non_zero_columns]
    ratio_all_ones = results['ratio_all_ones'][:, :nr_non_zero_columns]
    radius = results['radius'][:, :nr_non_zero_columns]
    variance_proba = results["variance_proba"][:, :nr_non_zero_columns]
    variance_logit = results.get('variance_logit', None)
    if variance_logit is not None:
        variance_logit = variance_logit[:, :nr_non_zero_columns]
    mse = results.get('mse', None)
    if mse is not None:
        mse = mse[:, :nr_non_zero_columns]
    mae = results.get('mae', None)
    if mae is not None:
        mae = mae[:, :nr_non_zero_columns]
    r2 = results.get('r2', None)
    if r2 is not None:
        r2 = r2[:, :nr_non_zero_columns]

    return (accuracy, precision, recall, f1,
            mse_proba, mae_proba, r2_proba,
            mse, mae, r2,
            gini, ratio_all_ones, variance_proba,
            variance_logit,
            radius), n_points_in_ball

