import numpy as np
import matplotlib.pyplot as plt
# from src.explanation_methods.lime_analysis.lime_local_classifier import get_feat_coeff_intercept
from matplotlib.lines import Line2D
from matplotlib import cm
from typing import Optional
from src.utils.process_results import load_and_get_non_zero_cols
from src.utils.metrics import weighted_avg_and_var
from src.explanation_methods.lime_analysis.lime_local_classifier import get_feat_coeff_intercept
import os
light_red = "#ffa5b3"
light_blue = "#9cdbfb"
light_grey = "#61cff2"
SYNTHETIC_DATASET_MAPPING = {
   "synthetic (simple)": "n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42",
    "synthetic (medium)":"n_feat50_n_informative10_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42",
     "synthetic (complex)":"n_feat100_n_informative50_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42"
}

def plot_weighted_average_metrics_vs_fractions(metric: list[np.array],
                                               models: list[str],
                                               k_nn: np.array,
                                               weights: np.array=None,
                                               plot_error_bars: bool=False,
                                               explanation_method: str="Integrated Gradients",
                                               y_lim: tuple=None,
                                               x_label: str="kNN of $x$",
                                               metrics_string: str="Accuracy",
                                               ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    weighted_ags_vars = [weighted_avg_and_var(metric_i, weights) for metric_i in metric]
    cmap = plt.get_cmap('tab10')
    for i, (model_str) in enumerate(models):
        color = cmap(i % 10)
        weighted_avg = weighted_ags_vars[0][i]
        weighted_var = weighted_ags_vars[1][i]
        if plot_error_bars:
            ax.plot(k_nn, weighted_avg, 'o-', color=color, label=model_str)
            ax.fill_between(k_nn, weighted_avg - np.sqrt(weighted_var), weighted_avg + np.sqrt(weighted_var), 
                           alpha=0.2, color=color)
        else:
            ax.plot(k_nn, weighted_avg,'o-', color=color, label=model_str)
    if weights is not None:
        ax.set_title(f'{explanation_method}: Weighted Average Fidelity over Test set of $f(x)$ vs. local explanation $g_{{x_0}}(x)$')
        ax.set_ylabel(f'Weighted Mean {metrics_string}', fontsize=12)
    else:
        ax.set_title(f'{explanation_method}: Average Fidelity over Test set of $f(x)$ vs. local explanation $g_{{x_0}}(x)$')
        ax.set_ylabel(f'Mean {metrics_string}', fontsize=12)
    ax.set_xlabel(x_label, fontsize=12)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=10, loc='upper right', ncol=2)
    if ax is None:
        plt.tight_layout()
        plt.show()
    return ax
    
def plot_metrics_per_sample(metric, neighbourhood_size, metric_str, ax = None, x_label="Fraction of Points in Ball"):
    if ax is None:
        fig, ax = plt.subplots()
    colors = cm.get_cmap('tab20', metric.shape[1])

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i in range(metric.shape[1]):
        ax.scatter(neighbourhood_size, metric[:, i], alpha=0.5, color=colors(i))
    ax.set_ylabel(metric_str)
    ax.set_xlabel(x_label)
    ax.set_title(f"{metric_str} per test sample")
    ax.grid(True)
    return ax

SYNTHETIC_DATASET_MAPPING = {
    "synthetic (simple)": "n_feat50_n_informative2_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class2_class_sep0.9_flip_y0.01_random_state42",
    "synthetic (medium)":"n_feat50_n_informative10_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42",
    "synthetic (complex)":"n_feat100_n_informative50_n_redundant30_n_repeated0_n_classes2_n_samples100000_n_clusters_per_class3_class_sep0.9_flip_y0.01_random_state42"
}

def plot_mean_metrics_per_model_and_dataset(results_files_dict, 
                                            metric_strs=["Accuracy", "MSE prob."], 
                                            distance_measure="euclidean",
                                            max_neighbours=None,
                                            plot_knn_metrics=True):
    metrics_str_to_idx = {
        "Accuracy": 0,
        "Precision": 1,
        "Recall": 2,
        "F1": 3,
        "MSE prob.": 4,
        "MAE prob.": 5,
        "R2  prob.": 6,
        "MSE logit": 7,
        "MAE logit": 8,
        "R2 logit": 9,
        "Gini Impurity": 10,
        "Ratio All Ones": 11,
        "Variance prob.": 12, 
        "Variance Logit": 13,
        "Radius": 14,
        "Accuraccy (vs. constant clf)": 15,
    }

    datasets = sorted(set(ds for model_data in results_files_dict.values() for ds in model_data.keys()))
    models = list(results_files_dict.keys())
    n_metrics = len(metric_strs)

    fig, axes = plt.subplots(len(datasets), n_metrics, figsize=(6 * n_metrics, 4 * len(datasets)))
    if len(datasets) == 1:
        axes = np.array([axes])  # Ensure axes is always a 2D array
    
    cmap = plt.get_cmap('tab10')
    model_lines = {}  # Store one line per model for a single legend

    for i, dataset in enumerate(datasets):
        for j, metric_str in enumerate(metric_strs):
            ax = axes[i, j]
            metric_idx = metrics_str_to_idx[metric_str]

            for k, model in enumerate(models):
                if dataset in results_files_dict[model]:
                    metrics, n_nearest_neighbours = load_and_get_non_zero_cols(results_files_dict[model][dataset])
                    metric = metrics[metric_idx]

                    if max_neighbours is None:
                        max_neighbours = len(n_nearest_neighbours)
                    metric = metric[:max_neighbours, :]
                    n_nearest_neighbours = n_nearest_neighbours[:max_neighbours]

                    if metric is not None and np.all(metric == None):
                        print(f"Warning: {model} - {dataset} - {metric_str} contains only None values")
                        continue

                    color = cmap(k % 10)
                    line, = ax.plot(n_nearest_neighbours, np.mean(metric, axis=1), '-', color=color, linewidth=2)
                    
                    if model not in model_lines:  
                        model_lines[model] = line  # Store for single legend

                    # Add reference horizontal lines for best kNN performance
                    if plot_knn_metrics:
                        synthetic = "synthetic" in dataset
                        dataset_str = dataset if not synthetic else SYNTHETIC_DATASET_MAPPING[dataset]
                        out = get_best_metrics_of_knn(model, dataset_str, metric_str, synthetic)
                        if out is not None:
                            ax.axhline(y=out[0], color=color, linestyle='--', alpha=0.7, label=f"{model} ({out[1]}-NN)")  

            ax.set_title(f"{dataset}: {metric_str}", fontsize=10)
            ax.set_xlabel("# Neighbors", fontsize=9)
            ax.set_ylabel(metric_str, fontsize=9)
            ax.grid(True, linestyle=':', linewidth=0.5)  # Subtle grid lines
            ax.tick_params(axis='both', which='major', labelsize=8)

    # Adjust layout for readability
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Single legend outside the plots
    fig.legend(model_lines.values(), model_lines.keys(), loc='upper center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f'Model Performance Metrics Across Different Datasets, distance metrics: {distance_measure}', fontsize=12, y=0.98)
    plt.show()



def get_best_metrics_of_knn(model, dataset, metric_sr, synthetic=False):
    metric_str_to_key_pair = {
        "Accuracy": ("classification", 0),
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
    if synthetic:
        # Get dataset name without synthetic_data/ prefix
        dataset_name = dataset.split('/')[-1]
        file_path = (f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/"
                    f"synthetic_data/{dataset_name}/kNN_on_model_preds_{model}_{dataset_name}_"
                    f"normalized_tensor_frame_dist_measure-euclidean_random_seed-42.npz")
    else:
        file_path = (f"/home/grotehans/xai_locality/results/knn_model_preds/{model}/"
                    f"{dataset}/kNN_on_model_preds_{model}_{model}_{dataset}_normalized_data_"
                    f"dist_measure-euclidean_random_seed-42.npz")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        res = np.load(file_path, allow_pickle=True)
        return np.max(res[metric_key_pair[0]][:, metric_key_pair[1]]), np.argmax(res[metric_key_pair[0]][:, metric_key_pair[1]])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None




def plot_mean_metrics_vs_fractions(means: list[np.array], fractions: np.array, kernels=None, explanation_method:str="LIME", y_lim:tuple=None, metrics_string:str="Accuracy", ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
    
    cmap = plt.get_cmap('tab10')
    for i, mean_accuracies in enumerate(means):
        color = cmap(i % 10)
        label = f'Kernel width: {kernels[i]}' if kernels is not None else None
        ax.scatter(fractions, mean_accuracies, color=color, s=50, marker='o', label=label)

    ax.set_title(f'{explanation_method}: Mean {metrics_string} vs. neighbourhood of $x$, $\\frac{{|N_d(x,*)|}}{{|D_{{test}}|}}$', fontsize=16)
    ax.set_xlabel('Relative Size of Approximation Region', fontsize=14)
    ax.set_ylabel(f'Mean Metric', fontsize=12)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if kernels is not None:
        ax.legend(fontsize=10, loc='upper right', ncol=2)  

    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    return ax

def plot_accuracy_vs_fraction(accuracy, 
                            fraction_points_in_ball, 
                            title_add_on="", 
                            save_path=None,
                            alpha=0.3):
    mean_accuracy = np.mean(accuracy, axis=1)
    color_array =  "#008080"

    # Create figure and axis objects
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    accuracy_t = accuracy.T.flatten()
    fraction_points_in_ball_t = fraction_points_in_ball.repeat(accuracy.shape[1])

    ax.scatter(fraction_points_in_ball_t, accuracy_t.flatten(), s=2, alpha = alpha, c=color_array)

    ax.scatter(fraction_points_in_ball, mean_accuracy, s=10, c='k', marker='x', label='Mean')
    ax.set_ylim(0, 1)

    ax.axhline(0.5, color='k', linestyle='dashed', linewidth=1)
    ax.set_xlabel("Relative Size of Approximation Region")
    ax.set_ylabel("Mean Accuracy") 
    ax.set_title(f"Accuracy vs. Relative Size of approximation region, {title_add_on}")

    legend_elements = [
        Line2D([0], [0], marker='x', color='k', linestyle='None', label='Mean', markersize=6)
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_accuracy_vs_threshold(accuracy, 
                                thresholds, 
                                model_predictions=None, 
                                title_add_on="",
                                save_path=None):
    mean_accuracy = np.mean(accuracy, axis=1)   
    
    light_violet = "#d8bfd8"
    light_red = "#ffa5b3"
    light_blue = "#9cdbfb"
    # Create figure and axis objects
    fig, ax = plt.subplots()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for dp in range(accuracy.shape[1]):
        if model_predictions is not None:
            color = light_red if model_predictions[dp] == 1 else light_blue
            ax.plot(thresholds, accuracy[:, dp], color=color, alpha=0.1)
    if model_predictions is None:
        color = light_violet
        ax.plot(thresholds, accuracy, color=color, alpha=0.1)
    ax.set_xlabel("Thresholds")
    # Plot mean accuracy
    ax.plot(thresholds, mean_accuracy, color='k', linestyle='dashed', linewidth=1, label="Mean accuracy")
    
    legend_elements = [Line2D([0], [0], linestyle='solid', color='k', linewidth=1, label='Mean accuracy')]
    if model_predictions is not None:
        legend_elements += [
            Line2D([0], [0], linestyle='solid', color='red', linewidth=1, label='Mean accuracy, pred: class 1'),
            Line2D([0], [0], linestyle='solid', color='blue', linewidth=1, label='Mean accuracy, pred: class 0')
        ]

    ax.legend(handles=legend_elements, loc='upper right')
    ax.axhline(0.5, color='k', linestyle='dashed', linewidth=1)
    ax.set_ylabel("Accuracy")
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_title(f"Accuracy per threshold {title_add_on}")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()





def plot_accuracy_vs_radius(accuracy, 
                            fraction_points_in_ball, 
                            model_predictions=None, 
                            kernel_ids = None, 
                            kernel_id_to_width=None,
                            title_add_on="", 
                            save_path=None,
                            alpha=0.03):
    mean_fraction = np.mean(fraction_points_in_ball, axis=1)
    mean_accuracy = np.mean(accuracy, axis=1)
    sem = np.std(accuracy, axis=1)
    light_red = "#ffa5b3"
    light_blue = "#9cdbfb"
    if model_predictions is not None:
        mean_accuracy_class_1 = np.mean(accuracy[:, model_predictions == 1], axis=1)
        mean_accuracy_class_0 = np.mean(accuracy[:, model_predictions == 0], axis=1)
        colors = [light_red if y == 1 else light_blue for y in model_predictions]
        color_array = np.repeat(colors, accuracy.T.shape[1], axis=0)
    if kernel_ids is not None:
        color_array = kernel_ids.T.flatten()
        # Map the numbers to a distinct color palette
        color_array = cm.Set1(kernel_ids.T.flatten())
        # Create a legend for the threshold colors
        unique_thresholds = np.unique(kernel_ids)
        kernel_ids_legend = [cm.Set1(thresh) for thresh in unique_thresholds]
        if kernel_id_to_width is not None:
            unique_thresholds = [kernel_id_to_width[k] for k in unique_thresholds]  
    else:
        color_array =  "#008080"

    # Create figure and axis objects
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if model_predictions is not None:
        ax.scatter(fraction_points_in_ball.T.flatten(), accuracy.T.flatten(), s=1.5, c=color_array)
        ax.scatter(mean_fraction, mean_accuracy_class_1, s=10, c='red', marker='x', label='Mean accuracy, pred: class 1')
        ax.scatter(mean_fraction, mean_accuracy_class_0, s=10, c='blue', marker='x', label='Mean accuracy, pred: class 0')
    else:
        ax.scatter(fraction_points_in_ball.T.flatten(), accuracy.T.flatten(), s=2, alpha = alpha, c=color_array)


    ax.scatter(mean_fraction, mean_accuracy, s=10, c='k', marker='x', label='Mean')
    # ax.errorbar(mean_fraction, mean_accuracy, yerr=sem, fmt='none', color='k', capsize=2)

    ax.axhline(0.5, color='k', linestyle='dashed', linewidth=1)
    ax.set_xlabel("Fraction of points in ball")
    ax.set_ylabel("Accuracy") 
    ax.set_title(f"Accuracy vs. Fraction of points in ball {title_add_on}")

    legend_elements = [
        Line2D([0], [0], marker='x', color='k', linestyle='None', label='Mean', markersize=6)
    ]
    if model_predictions is not None:
        legend_elements += [
            Line2D([0], [0], marker='x', color='red', linestyle='None', label='Mean accuracy, pred: 1', markersize=6),
            Line2D([0], [0], marker='x', color='blue', linestyle='None', label='Mean accuracy, pred: 0', markersize=6),
            Line2D([0], [0], marker='o', color=light_red, linestyle='None', label='Prediction: 1', markersize=6),
            Line2D([0], [0], marker='o', color=light_blue, linestyle='None', label='Prediction: 0', markersize=6)
        ]
    if kernel_ids is not None:
        legend_elements += [
            Line2D([0], [0], marker='o', color=kernel_ids_legend[i], linestyle='None', label=f'kernel width: {unique_thresholds[i]}', markersize=6)
            for i in range(len(unique_thresholds))
        ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_3d_scatter(fraction_points_in_ball, 
                    thresholds, 
                    accuracy, 
                    x_label="Fraction of points in ball", 
                    y_label="Thresholds", 
                    z_label="Accuracy", 
                    title="LIME Local Model on Test set", 
                    s=2, 
                    alpha=0.5, 
                    cmap='viridis', 
                    angles=(30, 40), 
                    save_path = None
                    ):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # mean_accuracy = np.mean(accuracy, axis=1)
    # mean_fraction = np.mean(fraction_points_in_ball, axis=1)

    x = fraction_points_in_ball.flatten()
    z = accuracy.flatten()
    y = np.repeat(thresholds, fraction_points_in_ball.shape[1])  # Flattened thresholds

    sc = ax.scatter(-x, y, z, c=z, s=s, alpha=alpha, cmap=cmap)  # Negated x to flip direction

    # # Plot mean values with negative x to match main scatter plot
    # ax.scatter(-mean_fraction, thresholds, mean_accuracy, c='r', s=10, marker='*', label='Mean')  # Added zorder to bring points to front
    # ax.legend()

    # Set labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)  # Add labelpad to move label further from axis
    ax.zaxis.labelpad=-0.7 # <- change the value here

    ax.set_title(title)

    # Set viewing angle (elevation, azimuth in degrees)
    ax.view_init(elev=angles[0], azim=angles[1])

    # Set x-axis limits and ticks
    ax.set_xlim(-1, 0)
    current_ticks = np.linspace(-1, 0, 6)
    ax.set_xticks(current_ticks)
    ax.set_xticklabels([f'{abs(x):.2f}' for x in current_ticks])

    # Add colorbar
    cb = fig.colorbar(sc, ax=ax, label=z_label)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.3)
    
    plt.show()


def plot_lime_stats(explanations, plot_intercepts=True, plot_means_coefficients=True, model_predictions=None, title_add_on= ""):
    means_coefficients = [np.mean(get_feat_coeff_intercept(exp)[1]) for i, exp in enumerate(explanations)]
    intercepts = [get_feat_coeff_intercept(exp)[2] for i, exp in enumerate(explanations)]
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    if plot_intercepts:
        # Plot vertical lines for intercepts
        colors = ['red' if y == 1 else 'blue' for y in model_predictions] if model_predictions is not None else ['blue'] * len(intercepts)
        ax1.scatter(range(len(intercepts)), intercepts, s=1, c=colors)
        ax1.set_title("LIME - Intercepts per test datapoint"+ " " + title_add_on)
        ax1.set_xlabel("Test datapoint")
        ax1.set_ylabel("Intercept")
        
        # Add legend
        if model_predictions is not None:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='red', label='Prediction: 1', markersize=8),
                             Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='blue', label='Prediction: 0', markersize=8)]
            ax1.legend(handles=legend_elements)

    if plot_means_coefficients:
        #plot histogram of the means_coefficients
        ax2.hist(means_coefficients, bins=100)
        ax2.set_title("LIME - Means of the coefficients per test datapoint")
        ax2.set_xlabel("Mean of the coefficients")
        ax2.set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
