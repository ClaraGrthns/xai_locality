import os
import numpy as np
import re
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os.path as osp

DATASET_TO_NUM_FEATURES = {"higgs": 24,
                           "jannis": 54,
                           "synthetic_data/n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state{random_seed}": 50, 
            "synthetic_data/n_feat50_n_informative10_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state{random_seed}": 50,
            "synthetic_data/n_feat100_n_informative50_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state{random_seed}": 100,
}


def file_matching(file, distance_measure, condition=lambda x: True):
        return condition(file) and distance_measure in file


def get_kernel_widths_to_filepaths(files):
    """Extract kernel widths from file paths."""
    if not isinstance(files, list):
        files = [files]
    widths = []
    for f in files:
        match = re.search(r'kernel_width-(\d+\.?\d*)', str(f))
        widths.append((float(match.group(1)), f) if match else (None, f))
    return sorted(widths, key=lambda x: x[0])

def get_synthetic_dataset_mapping(datasets, regression=False):
    """Generate a mapping between user-friendly names and full synthetic dataset names"""
    mapping = {}
    if type(datasets) == str:
        datasets = [datasets]
    for dataset in datasets:
        if 'syn' in dataset:
            if regression:
                friendly_name = get_synthetic_dataset_friendly_name_regression(dataset)
            else:
                friendly_name = get_synthetic_dataset_friendly_name(dataset)
            mapping[friendly_name] = dataset
    return mapping

def get_synthetic_dataset_friendly_name(dataset_name, pattern=None):
    """Generate a user-friendly name for synthetic datasets using regex to extract parameters"""
    if pattern is None:
        pattern = r'n_feat(\d+)_n_informative(\d+).*?n_clusters_per_class(\d+).*?class_sep([\d\.]+)'
    match = re.search(pattern, dataset_name.split("/")[-1])
    if match:
        d = match.group(1)       # number of features
        i = match.group(2)       # number of informative features
        c = match.group(3)       # clusters per class
        s = match.group(4)       # class separation
        hypercube_param = "hc: ×" if "hypercubeFalse" in dataset_name else "hc: ✓"
        return f"syn (d:{d}, if:{i}, c:{c}, s:{s}, {hypercube_param})"
    return dataset_name

def get_synthetic_dataset_friendly_name_regression(dataset_name, pattern=None):
    """Generate a user-friendly name for synthetic regression datasets using regex to extract parameters"""
    if pattern is None:
        pattern = r'regression_(\w+)_n_feat(\d+)_n_informative(\d+)_n_samples(\d+)_noise([\d\.]+)_bias([\d\.]+)(?:_random_state(\d+))?(?:_effective_rank(\d+)_tail_strength([\d\.]+))?'
    match = re.search(pattern, dataset_name.split("/")[-1])
    if match:
        mode = match.group(1)    # regression mode (e.g., 'linear', 'friedman')
        d = match.group(2)       # number of features
        i = match.group(3)       # number of informative features
        n = match.group(4)       # number of samples
        noise = match.group(5)   # noise level
        bias = match.group(6)    # bias
        # Check for additional parameters
        additional = ""
        # Extract effective rank number if present
        effective_rank_match = re.search(r'effective_rank(\d+)', dataset_name)
        if effective_rank_match:
            rank_num = effective_rank_match.group(1)
            additional += f", er:{rank_num}"
        return f"syn-reg {mode} \n(d:{d}, if:{i}, b: {bias}, ns:{noise}{additional})"
    return dataset_name

def get_results_files_dict(explanation_method: str,
                        models: list[str], 
                        datasets: list[str], 
                        distance_measure: str = "euclidean", 
                        lime_features=10, 
                        sampled_around_instance=False, 
                        random_seed=42,
                        downsample_analysis=1.0) -> dict:
    results_folder = f"/home/grotehans/xai_locality/results/{explanation_method}"
    results_files_dict = {}
    if type(models) == str:
        models = [models]
    if type(datasets) == str:
        datasets = [datasets]
    for model in models:
        results_files_dict[model] = {}
        for dataset in datasets:
            path_to_results = os.path.join(results_folder, model, dataset)
            if not os.path.exists(path_to_results):
                continue
            if sampled_around_instance:
                condition = lambda x: x.startswith("sampled") and x.endswith("fraction.npz")
            else:
                if explanation_method == "lime":
                    if isinstance(lime_features, int) and lime_features != 10:
                        condition = lambda x: x.startswith("fractions") and f"num_features-{lime_features}.npz" in x
                    elif lime_features == "all":
                        num_feat = DATASET_TO_NUM_FEATURES[dataset]
                        condition = lambda x: (x.startswith("fractions") or x.startswith("regression")) and f"num_features-{num_feat}.npz" in x
                    else:
                        condition = lambda x: (x.startswith("fractions") or x.startswith("regression")) and x.endswith("fraction.npz")
                else:
                    condition = lambda x: (x.startswith("fractions") or x.startswith("regression")) and x.endswith("fraction.npz")
            
            def random_seed_condition(file):
                if random_seed == 42:
                    return f"random_seed-{random_seed}" in file or "random_seed" not in file
                return f"random_seed-{random_seed}" in file
            
            def downsample_condition(file):
                if downsample_analysis == 1.0:
                    return "downsample" not in file
                else:
                    return f"downsample-{np.round(downsample_analysis, 2)}" in file
            # Combine conditions
            combined_condition = lambda x: file_matching(x, distance_measure, condition=condition) and random_seed_condition(x) and downsample_condition(x)
            
            res = [os.path.join(path_to_results, f) for f in os.listdir(path_to_results) if combined_condition(f)]
            if explanation_method == "lime":
                results_files_dict[model][dataset] = res
            elif len(res) > 0:
                results_files_dict[model][dataset] = res[0]
    # Rename synthetic dataset keys
    for model in results_files_dict:
        keys = list(results_files_dict[model].keys())
        for key in keys:
            if "regression_synthetic_data" in key:
                friendly_name = get_synthetic_dataset_friendly_name_regression(key)
                results_files_dict[model][friendly_name] = results_files_dict[model].pop(key)
            elif "synthetic_data/" in key:
                friendly_name = get_synthetic_dataset_friendly_name(key)
                results_files_dict[model][friendly_name] = results_files_dict[model].pop(key)
    return results_files_dict

def get_non_zero_cols(array):
    return array.shape[1] - np.sum(np.all(array == 0, axis=0))


def load_results_clf(data_path):
    """
    Load results from a numpy file and extract non-zero columns for various metrics.

    Parameters:
    data_path (str): Path to the numpy file containing the results.

    Returns:
    tuple: A tuple containing:
        - A tuple of numpy arrays for the following metrics, each truncated to non-zero columns: shapes: kNN x testpoints
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
    # nr_non_zero_columns = get_non_zero_cols(results['accuracy']) 
    n_points_in_ball = results['n_points_in_ball']
    # Common metrics for both methods
    accuraccy_constant_clf = results['accuraccy_constant_clf']
    accuracy = results['accuracy']
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1']
    mse_proba = results['mse_proba']
    mae_proba = results['mae_proba']
    r2_proba = results['r2_proba']
    gini = results["gini"]
    ratio_all_ones = results['ratio_all_ones']
    radius = results['radius']
    variance_proba = results["variance_proba"]
    variance_logit = results.get('variance_logit', None)
    if variance_logit is not None:
        variance_logit = variance_logit
    mse = results.get('mse', None)
    if mse is not None:
        mse = mse
    mae = results.get('mae', None)
    if mae is not None:
        mae = mae
    r2 = results.get('r2', None)
    if r2 is not None:
        r2 = r2
    ratio_all_ones_local = results.get("ratio_all_ones_local", None)
    if ratio_all_ones_local is not None:
        ratio_all_ones_local = ratio_all_ones_local

    return (accuracy, precision, recall, f1,
            mse_proba, mae_proba, r2_proba,
            mse, mae, r2,
            gini, ratio_all_ones, variance_proba,
            variance_logit,
            radius,
            accuraccy_constant_clf,
            ratio_all_ones_local
            ), n_points_in_ball

def load_results_regression(data_path):
    """
    Load results from a numpy file and extract non-zero columns for various metrics.

    Parameters:
    data_path (str): Path to the numpy file containing the results.

    Returns:
    tuple: A tuple containing:
        - A tuple of numpy arrays for the following metrics, each truncated to non-zero columns: shapes: kNN x testpoints
            - mse (0)
            - mae (1)
            - r2 (2)
            - mse_constant_clf (3)
            - mae_constant_clf (4)
            - variance_logit (5)
            - radius (6)
        - n_points_in_ball (numpy array): Number of points in the ball.
    """
    results = np.load(data_path, allow_pickle=True)
    # nr_non_zero_columns = get_non_zero_cols(results['mse']) 
    n_points_in_ball = results['n_points_in_ball']
    # Extract metrics for regression tasks
    n_points_in_ball = results['n_points_in_ball']
    
    # Extract the metrics needed for regression analysis
    mse = results['mse']
    mae = results['mae']
    r2 = results['r2']
    mse_constant_clf = results['mse_constant_clf']
    mae_constant_clf = results['mae_constant_clf']
    variance_logit = results.get('variance_logit', None)
    if variance_logit is not None:
        variance_logit = variance_logit
    radius = results['radius']

    return (mse, mae, r2,
            mse_constant_clf, mae_constant_clf,
            variance_logit, radius), n_points_in_ball


def load_knn_results(model, dataset, synthetic=False, distance_measure="euclidean", regression=False, random_seed=42, downsample_analysis=1.0):
    distance_measure = distance_measure.lower()
    suffix = "_regression" if regression else ""
    prefix = "regression_" if regression else ""
    if synthetic:
        # Get dataset name without synthetic_data/ prefix
        dataset_name = dataset.split('/')[-1]
        file_path = (f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/"
                    f"{prefix}synthetic_data/{dataset_name}/kNN{suffix}_on_model_preds_{model}_{dataset_name}_"
                    f"normalized_tensor_frame_dist_measure-{distance_measure}_random_seed-{random_seed}{f"downsample-{np.round(downsample_analysis, 2)}" if downsample_analysis!=1.0 else ""}.npz")
    else:
        if model == "LogReg" or model == "LinReg":
            file_path = (f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/"
                    f"{dataset}/kNN{suffix}_on_model_preds_{model}_LightGBM_{dataset}_normalized_data_"
                    f"dist_measure-{distance_measure}_random_seed-{random_seed}{f"downsample-{np.round(downsample_analysis, 2)}" if downsample_analysis!=1.0 else ""}.npz")
        else: 
            file_path = (f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/"
                    f"{dataset}/kNN{suffix}_on_model_preds_{model}_{model}_{dataset}_normalized_data_"
                    f"dist_measure-{distance_measure}_random_seed-{random_seed}{f"downsample-{np.round(downsample_analysis, 2)}" if downsample_analysis!=1.0 else ""}.npz")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        res = np.load(file_path, allow_pickle=True)
        return res
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None   

def load_model_performance(model, dataset, synthetic=False, random_seed=42):
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
        file_path = f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/synthetic_data/{dataset_name}/model_performance_{model}_{dataset_name}_normalized_tensor_frame_random_seed-{random_seed}.npz"
    else:
        if model == "LogReg":
            file_path = f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/{dataset}/model_performance_{model}_LightGBM_{dataset}_normalized_data_random_seed-{random_seed}.npz"
        else:
            file_path = f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/{dataset}/model_performance_{model}_{model}_{dataset}_normalized_data_random_seed-{random_seed}.npz"
    try:
        res = np.load(file_path, allow_pickle=True)
        return res
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
def get_performance_metrics_model(model, dataset, metric_str, synthetic=False, random_seed=42):
    res = load_model_performance(model, dataset, synthetic, random_seed)
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


def get_best_metrics_of_knn_clf(model, dataset, metric_sr_ls, synthetic=False, distance_measure="euclidean", random_seed=42):
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
        "R2 logit": ("logit_regression", 2),
        "Accuracy true labels": ("classification_true_labels", 0),
        "Precision true labels": ("classification_true_labels", 1),
        "Recall true labels": ("classification_true_labels", 2),
        "F1 true labels": ("classification_true_labels", 3),
    }
    if type(metric_sr_ls) == str:
        metric_sr_ls = [metric_sr_ls]
    res = load_knn_results(model = model, 
                           dataset = dataset, 
                           synthetic = synthetic, 
                           distance_measure= distance_measure, 
                           random_seed = random_seed, 
                           regression=False)
    if res is None:
        return None
    metrics_res = []
    for metric_sr in metric_sr_ls:
        if metric_sr not in metric_str_to_key_pair:
            metrics_res.append((np.nan, np.nan))
            continue
        metric_key_pair = metric_str_to_key_pair[metric_sr]
        best_metric = np.max(res[metric_key_pair[0]][:, metric_key_pair[1]])
        best_idx = np.argmax(res[metric_key_pair[0]][:, metric_key_pair[1]])+1
        metrics_res.append((best_metric, best_idx))
    return metrics_res if len(metrics_res) > 1 else metrics_res[0]

def get_best_metrics_of_knn_regression(model, dataset, metric_sr_ls, synthetic=False, distance_measure="euclidean", random_seed=42):
    distance_measure = distance_measure.lower()
    metric_str_to_key_pair = {
        "MSE $g_x$": ("res_regression", 0),
        "RMSE $g_x$": ("res_regression", 0),
        "MAE $g_x$": ("res_regression", 1),
        "R2 $g_x$": ("res_regression", 2),
        "MSE true labels": ("res_regression_true_y", 0),
        "MAE true labels": ("res_regression_true_y", 1),
        "R2 true labels": ("res_regression_true_y", 2),
    }
    if type(metric_sr_ls) == str:
        metric_sr_ls = [metric_sr_ls]

    res = load_knn_results(model=model, 
                           dataset=dataset, synthetic = synthetic, 
                           distance_measure= distance_measure, 
                           random_seed = random_seed,  
                           regression = True,
                        )
    if res is None:
        return None
    metrics_res = []
    for metric_sr in metric_sr_ls:
        if metric_sr not in metric_str_to_key_pair:
            metrics_res.append((np.nan, np.nan))
            continue
        metric_key_pair = metric_str_to_key_pair[metric_sr]
        best_metric = np.max(res[metric_key_pair[0]][:, metric_key_pair[1]])
        if metric_sr == "RMSE $g_x$":
            best_metric = np.sqrt(best_metric)
        best_idx = np.argmax(res[metric_key_pair[0]][:, metric_key_pair[1]])+1
        metrics_res.append((best_metric, best_idx))
    return metrics_res if len(metrics_res) > 1 else metrics_res[0]

