import argparse
import numpy as np
from utils.plotting_utils import plot_accuracy_vs_fraction
from model.lightgbm import load_model, load_data, predict_fn
from functools import partial
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description="Plot accuracy vs fraction for different kernel widths.")
    parser.add_argument('--kernel_widths', nargs='+', type=float, required=True, help='List of kernel widths.')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory to save results.')
    parser.add_argument('--setting', type=str, required=True, help='Setting string.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data.')
    parser.add_argument("--data_folder", type=str, help="Path to the data folder")
    parser.add_argument("--model_folder", type=str, help="Path to the model folder")
    parser.add_argument("--results_folder", type=str, help="Path to the results folder")
    parser.add_argument("--model_type", type=str, default="lightgbm", help="Model type, so far only 'xgboost' and 'lightgbm' is supported")
    parser.add_argument("--results_path", type=str, help="Path to save results", default="/home/grotehans/xai_locality/results/lightgbm/jannis")
    parser.add_argument("--distance_measure", type=str, default="euclidean", help="Distance measure")
    parser.add_argument("--experiment_setting_threshold", type=str, help="Leave kernel_width as {kw}")
    parser.add_argument("--num_tresh", type=int, default=150, help="Number of thresholds")
    parser.add_argument("--model_regressor", type=str, default="ridge", help="Model regressor for LIME")
    return parser.parse_args()

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

def get_non_zero_cols(array):
    return array.shape[1] - np.sum(np.all(array == 0, axis=0))

def load_and_get_non_zero_cols(data_path):
    accuracy_array, fraction_array, thresholds, _, _, _ = load_results(data_path)
    non_zero_cols = get_non_zero_cols(fraction_array)
    print(f"computed up to {non_zero_cols} data points")

    accuracy_complete = accuracy_array[:, :non_zero_cols]
    fraction_complete = fraction_array[:, :non_zero_cols]
    return accuracy_complete, fraction_complete, thresholds, non_zero_cols

def main():
    args = parse_args()
    def get_path(base_folder, base_path, setting, suffix=""):
        if base_folder is None:
            return base_path
        assert setting is not None, "Setting must be specified if folder is provided"
        return osp.join(base_folder, f"{suffix}{setting}")
    
    results_path = get_path(args.results_folder, args.results_path, args.setting)
    result_paths = [osp.join(results_path, args.experiment_setting.format(kw)) for kw in args.kernel_widths]
    accuracy_arrays = []
    fraction_arrays = []
    non_zero_cols_list = []

    for result_path in result_paths:
        accuracy_array, fraction_array, _, non_zero_cols = load_and_get_non_zero_cols(result_path)
        accuracy_arrays.append(accuracy_array)
        fraction_arrays.append(fraction_array)
        non_zero_cols_list.append(non_zero_cols)

    min_cols = min(non_zero_cols_list)
    accuracies_concat = np.stack([acc[:, :min_cols] for acc in accuracy_arrays], axis=0)
    fraction_concat = np.stack([frac[:, :min_cols] for frac in fraction_arrays], axis=0)

    max_accuracies = np.max(accuracies_concat, axis=0)
    max_indices = np.argmax(accuracies_concat, axis=0)
    selected_fractions = fraction_concat[max_indices, np.arange(fraction_concat.shape[1])[:, None], np.arange(fraction_concat.shape[2])]
    selected_fractions.shape

    kernel_id_to_width = {i: str(kw) for i, kw in enumerate(args.kernel_widths)}
    plot_accuracy_vs_fraction(max_accuracies, selected_fractions, kernel_ids=max_indices, kernel_id_to_width=kernel_id_to_width, title_add_on=" - max over all thresholds")

if __name__ == "__main__":
    main()
