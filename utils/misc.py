import random
import numpy as np
import torch

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_results(data_path):
    results = np.load(data_path, allow_pickle=True)
    accuracy_array = results['accuracy']
    fraction_array = results['fraction_points_in_ball']
    thresholds = results['thresholds']
    n_samples_in_ball = results['samples_in_ball']
    ratio_all_ones = results['ratio_all_ones']
    return accuracy_array, fraction_array, thresholds, n_samples_in_ball, ratio_all_ones

def get_non_zero_cols(array):
    return array.shape[1] - np.sum(np.all(array == 0, axis=0))

def load_and_get_non_zero_cols(data_path):
    print(f"loading data from {data_path}")
    accuracy_array, fraction_array, thresholds, _, _= load_results(data_path)
    non_zero_cols = get_non_zero_cols(fraction_array)
    print(f"computed up to {non_zero_cols} data points")

    accuracy_complete = accuracy_array[:, :non_zero_cols]
    fraction_complete = fraction_array[:, :non_zero_cols]
    return accuracy_complete, fraction_complete, thresholds, non_zero_cols


def get_path(base_folder, base_path, setting, suffix=""):
    if base_folder is None:
        return base_path
    assert setting is not None, "Setting must be specified if folder is provided"
    return osp.join(base_folder, f"{suffix}{setting}")
