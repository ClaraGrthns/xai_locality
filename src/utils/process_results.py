import os
import numpy as np

DATASET_TO_NUM_FEATURES = {"higgs": 24,
                           "jannis": 54,
                           "synthetic_data/n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42": 50, 
            "synthetic_data/n_feat50_n_informative10_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42": 50,
            "synthetic_data/n_feat100_n_informative50_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42": 100,
}


def file_matching(file, distance_measure, condition=lambda x: True):
        return condition(file) and distance_measure in file
        
def get_results_files_dict(explanation_method: str, models: list[str], datasets: list[str], distance_measure:str="euclidean", lime_features = 10) -> dict:
    results_folder = f"/home/grotehans/xai_locality/results/{explanation_method}"
    results_files_dict = {}
    for model in models:
        results_files_dict[model] = {}
        for dataset in datasets:
            path_to_results = os.path.join(results_folder, model, dataset)
            if not os.path.exists(path_to_results):
                continue
            if isinstance(lime_features, int) and lime_features != 10:
                condition = lambda x: f"num_features-{lime_features}.npz" in x
            elif lime_features == "all":
                num_feat = DATASET_TO_NUM_FEATURES[dataset]
                condition = lambda x: f"num_features-{num_feat}.npz" in x
            else:
                condition = lambda x: x.endswith("fraction.npz")
            res = [os.path.join(path_to_results, f) for f in os.listdir(f"{results_folder}/{model}/{dataset}") if file_matching(f, distance_measure, condition=condition)]
            if len(res) > 0 and explanation_method == "lime":
                results_files_dict[model][dataset] = res
            elif len(res) > 0:
                results_files_dict[model][dataset] = res[0]
    
    # Rename synthetic dataset keys
    for model in results_files_dict:
        keys = list(results_files_dict[model].keys())
        for key in keys:
            if "synthetic_data/n_feat50_n_informative2" in key:
                results_files_dict[model]["synthetic (d:50, inf feat.: 2)"] = results_files_dict[model].pop(key)
            elif "synthetic_data/n_feat50_n_informative10" in key:
                results_files_dict[model]["synthetic (d:50, inf feat.: 10)"] = results_files_dict[model].pop(key)
            elif "synthetic_data/n_feat100" in key:
                results_files_dict[model]["synthetic (d:100, inf feat.: 50)"] = results_files_dict[model].pop(key)
                
    return results_files_dict

def get_non_zero_cols(array):
    return array.shape[1] - np.sum(np.all(array == 0, axis=0))


def load_and_get_non_zero_cols(data_path):
    """
    Load results from a numpy file and extract non-zero columns for various metrics.

    Parameters:
    data_path (str): Path to the numpy file containing the results.

    Returns:
    tuple: A tuple containing:
        - A tuple of numpy arrays for the following metrics, each truncated to non-zero columns:
            - accuracy (0)
            - precision (1)
            - recall (2)
            - f1 (3)
            - mse_proba (4)
            - mae_proba (5)
            - r2_proba (6)
            - mse (7, if available else None)
            - mae (8, if available else None)
            - r2 (9, if available else None)
            - gini (10)
            - ratio_all_ones (11)
            - variance_proba (12)
            - variance_logit (13, if available else None)
            - radius (14)
            - accuraccy_constant_clf (15)
            - ratio_all_ones_local (16, if available else None)
        - n_points_in_ball (numpy array): Number of points in the ball.
    """
    results = np.load(data_path, allow_pickle=True)
    nr_non_zero_columns = get_non_zero_cols(results['accuracy']) 
    n_points_in_ball = results['n_points_in_ball']
    # Common metrics for both methods
    accuraccy_constant_clf = results['accuraccy_constant_clf'][:, :nr_non_zero_columns]
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
    ratio_all_ones_local = results.get("ratio_all_ones_local", None)
    if ratio_all_ones_local is not None:
        ratio_all_ones_local = ratio_all_ones_local[:, :nr_non_zero_columns]

    return (accuracy, precision, recall, f1,
            mse_proba, mae_proba, r2_proba,
            mse, mae, r2,
            gini, ratio_all_ones, variance_proba,
            variance_logit,
            radius,
            accuraccy_constant_clf,
            ratio_all_ones_local
            ), n_points_in_ball



def load_knn_results(model, dataset, synthetic=False, distance_measure="euclidean"):
    distance_measure = distance_measure.lower()
    if synthetic:
        # Get dataset name without synthetic_data/ prefix
        dataset_name = dataset.split('/')[-1]
        file_path = (f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/"
                    f"synthetic_data/{dataset_name}/kNN_on_model_preds_{model}_{dataset_name}_"
                    f"normalized_tensor_frame_dist_measure-{distance_measure}_random_seed-42.npz")
    else:
        if model == "LogisticRegression":
            file_path = (f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/"
                    f"{dataset}/kNN_on_model_preds_{model}_LightGBM_{dataset}_normalized_data_"
                    f"dist_measure-{distance_measure}_random_seed-42.npz")
        else: 
            file_path = (f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/"
                    f"{dataset}/kNN_on_model_preds_{model}_{model}_{dataset}_normalized_data_"
                    f"dist_measure-{distance_measure}_random_seed-42.npz")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        res = np.load(file_path, allow_pickle=True)
        return res
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None   

def load_model_performance(model, dataset, synthetic=False):
    """Loads model performance metrics from a .npz file.

    Parameters
    ----------
    model : str
        Name of the model
    dataset : str
        Name of the dataset
    synthetic : bool, optional
        Whether the dataset is synthetic, by default False

    Returns
    -------
    numpy.lib.npyio.NpzFile
        NPZ file containing model performance metrics with key 'classification_model',
        where the array contains [accuracy, precision, recall, f1] scores
    """
    if synthetic:
        dataset_name = dataset.split('/')[-1]
        file_path = f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/synthetic_data/{dataset_name}/model_performance_{model}_{dataset_name}_normalized_tensor_frame_random_seed-42.npz"
    else:
        if model == "LogisticRegression":
            file_path = f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/{dataset}/model_performance_{model}_LightGBM_{dataset}_normalized_data_random_seed-42.npz"
        else:
            file_path = f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/{dataset}/model_performance_{model}_{model}_{dataset}_normalized_data_random_seed-42.npz"
    try:
        res = np.load(file_path, allow_pickle=True)
        return res
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
def get_performance_metrics_model(model, dataset, metric_str, synthetic=False):
    res = load_model_performance(model, dataset, synthetic)
    if res is None:
        return np.nan
    else:
        metric_str_to_key_pair = {
            "AUROC": 0,
            "Accuracy": 1,
            "Precision": 2,
            "Recall": 3,
            "F1": 4
        }
        return float(res['classification_model'][metric_str_to_key_pair[metric_str]])

def get_best_metrics_of_knn(model, dataset, metric_sr, synthetic=False, distance_measure="euclidean"):
    distance_measure = distance_measure.lower()
    metric_str_to_key_pair = {
        "Accuracy $g_x$": ("classification", 0),
        "Precision": ("classification", 1),
        "Recall": ("classification", 2),
        "F1": ("classification", 3),
        "MSE prob.": ("proba_regression", 0),
        "MAE prob.": ("proba_regression", 1),
        "R2  prob.": ("proba_regression", 2),
        "MSE logit": ("logit_regression", 0),
        "MAE logit": ("logit_regression", 1),
        "R2 logit": ("logit_regression", 2)
    }
    if metric_sr not in metric_str_to_key_pair:
        return None
    metric_key_pair = metric_str_to_key_pair[metric_sr]
    res = load_knn_results(model, dataset, synthetic, distance_measure)
    if res is None:
        return None
    else:
        return np.max(res[metric_key_pair[0]][:, metric_key_pair[1]]), np.argmax(res[metric_key_pair[0]][:, metric_key_pair[1]])   
