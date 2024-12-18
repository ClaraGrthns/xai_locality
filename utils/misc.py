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

def get_non_zero_cols(array):
    return array.shape[1] - np.sum(np.all(array == 0, axis=0))


def load_results(data_path):
    results = np.load(data_path, allow_pickle=True)
    print(results.files)
    accuracy_array = results['accuracy']
    fraction_array = results['fraction_points_in_ball']
    thresholds = results['thresholds']
    n_samples_in_ball = results['samples_in_ball']
    ratio_all_ones = results['ratio_all_ones']
    radius = results['radius']
    return accuracy_array, fraction_array, thresholds, n_samples_in_ball, ratio_all_ones, radius

    return array.shape[1] - np.sum(np.all(array == 0, axis=0))