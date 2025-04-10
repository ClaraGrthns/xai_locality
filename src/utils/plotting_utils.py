import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import gridspec, lines
from collections import defaultdict

from src.utils.process_results import get_results_files_dict, get_kernel_widths_to_filepaths, get_synthetic_dataset_mapping
# Constants
METRICS_TO_IDX_CLF = {
    "Accuracy $g_x$": 0,
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
    "Accuracy const. local model": 11,
    "Variance prob.": 12, 
    "Variance logit": 13,
    "Radius": 14,
    "Local Ratio All Ones": 16,
    "Accuracy $g_x$ - Accuracy const. local model": (0, 11),
}

METRICS_MAP_REG = {
    "MSE $g_x$": 0,
    "MAE $g_x$": 1,
    "R2 $g_x$": 2,
    "MSE const. local model": 3,
    "MAE const. local model": 4,
    "Variance $f(x)$": 5,
    "Radius": 6,
    "MSE const. local model - MSE $g_x$": (3, 0),
    "MAE const. local model - MAE $g_x$": (4, 1)
}


LINESTYLES = [  "--","-", "-."]
KERNEL_LABELS = ["default/2", "default", "default*2"]

def plot_metric(ax, values, neighbors, color, style, max_neighbors=None):
    """Plot a single metric line."""
    if values is None:
        print("Warning: Missing values")
        return
    
    neighbors = neighbors[:max_neighbors]
    values = values[:max_neighbors]
    ax.plot(neighbors, np.mean(values, axis=1), color=color, 
           linestyle=style, linewidth=2)

def setup_plot(n_metrics):
    """Initialize figure with subplots and legend space."""
    fig = plt.figure(figsize=(6 * n_metrics + 2, 4))
    gs = gridspec.GridSpec(1, n_metrics + 1, width_ratios=[6] * n_metrics + [1.5])  # Increase legend space
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_metrics)]
    legend_ax = fig.add_subplot(gs[0, -1])
    legend_ax.axis('off')  
    fig.subplots_adjust(wspace=0.4)  
    return fig, axes, legend_ax

def create_legend(models, colors, method, unique_kw_lines_idx):
    """Generate legend handles and labels."""
    handles = []
    labels = []
    handles.append(lines.Line2D([], [], color='none'))
    labels.append('Explanation for Models:')
    for m in models:
        handles.append(lines.Line2D([], [], color=colors[m], linewidth=2))
        labels.append(m)
    if method == "lime":
        handles.append(lines.Line2D([], [], color='none'))
        labels.append('Kernel Width:')
        for i in unique_kw_lines_idx:
            handles.append(lines.Line2D([], [], color='k', linestyle=LINESTYLES[i], linewidth=2))
            labels.append(KERNEL_LABELS[i])
    return handles, labels

def plot_dataset_metrics(models, datasets, method, metrics, distance="euclidean", 
                       max_neighbors=None, save=False, lime_features=10, regression=False, random_seed=42):
    """Main plotting function."""
    if regression:
        from src.utils.process_results import load_results_regression as load_results
    else:
        from src.utils.process_results import load_results_clf as load_results
    results = get_results_files_dict(method, models, datasets, distance, lime_features, random_seed=random_seed)
    cmap = plt.cm.tab10
    if regression:
        metrics_map = METRICS_MAP_REG
    else:
        metrics_map = METRICS_TO_IDX_CLF
    for dataset in set(d for model_data in results.values() for d in model_data):
        fig, axes, legend_ax = setup_plot(len(metrics))
        dataset_models = [m for m in results if dataset in results[m]]
        colors = {m: cmap(i) for i, m in enumerate(dataset_models)}
        unique_kw_lines = set()
        for ax, metric in zip(axes, metrics):
            
            metric_idx = metrics_map[metric]
            is_diff = "-" in metric or "Difference" in metric
            for model in dataset_models:
                files = results[model][dataset]
                if method == "lime" and metric not in ["Variance $f(x)$", "Radius"]:
                    kernel_widths_fp = get_kernel_widths_to_filepaths(files)
                    default_kw = np.median([kw[0] for kw in kernel_widths_fp])
                    for i, (width, path) in enumerate(kernel_widths_fp):
                        data, neighbors = load_results(path)
                        neighbors = np.arange(0, len(neighbors))
                        neighbors = neighbors + 1 if neighbors[0] == 0 else neighbors
                        vals = data[metric_idx[0]] - data[metric_idx[1]] if is_diff else data[metric_idx]
                        idx_linestyle = int(default_kw == width) + int(width > default_kw) - int(width < default_kw)
                        unique_kw_lines.add(idx_linestyle)
                        plot_metric(ax, vals, neighbors, colors[model], LINESTYLES[idx_linestyle], max_neighbors)
                else:
                    data, neighbors = load_results(files[0] if isinstance(files, list) else files)
                    neighbors = neighbors + 1 if neighbors[0] == 0 else neighbors
                    vals = data[metric_idx[0]] - data[metric_idx[1]] if is_diff else data[metric_idx]
                    plot_metric(ax, vals, neighbors, colors[model], '-', max_neighbors)
                
                if is_diff:
                    ax.axhline(0, color='k', alpha=0.8, linewidth=0.8)
            ax.set_title(metric)
            ax.set_xlabel(f"Neighborhood size ({distance} distance)")
            ax.grid(True, linestyle=':')
        
        handles, labels = create_legend(dataset_models, colors, method, unique_kw_lines)
        legend_ax.legend(handles, labels, frameon=True, fontsize=9)
        method_title = method.split("/")[-1]
        method_title = " ".join(method_title.split("_"))
        if method == "lime" and lime_features == "all":
            method_title = method_title + " (all features)"

        title = f"{method_title.capitalize()} on {dataset.capitalize()}"
        if method == "lime" and lime_features == "all":
            title += " (all features)"
        fig.suptitle(title, fontsize=14, y=1.02)
        
        if save:
            plt.savefig(f"graphics/{method}_{dataset}_{'_'.join(metrics)}.pdf")
        plt.show()



def extract_sort_keys(dataset, regression = False):
    """Extract sorting keys from dataset name using regex."""
    if regression: 
        match = re.search(r'regression_\w+_n_feat(\d+)_n_informative(\d+)', dataset)
        return (int(match.group(1)), int(match.group(2))) if match else (float('inf'), float('inf'))
    else:
        pattern = r'd:(\d+).*?if:(\d+).*?c:(\d+).*?s:([\d\.]+)'
        match = re.search(pattern, dataset)
        if match:
            n_feat = int(match.group(1))
            inf_feat = int(match.group(2))
            n_clusters = int(match.group(3))
            class_sep = float(match.group(4))
            return n_feat, inf_feat,  n_clusters, -class_sep # Negative class_sep for descending order
        return float('inf'), float('inf'), float('inf'),0  # Default for non-matching datasets


def create_dual_legend(ax, datasets, colors, models, markers):
    """Create a combined legend for datasets and models."""
    # Dataset legend
    legend_elements = [
        plt.Line2D([], [], color='none', label='Datasets:', markersize=0),
        *[plt.Line2D([], [], marker='o', color=colors[d], label=d, linestyle='None') 
          for d in datasets],
        plt.Line2D([], [], color='none', label=''),
        plt.Line2D([], [], color='none', label='Models:', markersize=0),
        *[plt.Line2D([], [], marker=markers[i], color='gray', label=m, linestyle='None') 
          for i, m in enumerate(models)]
    ]
    legend = ax.legend(handles=legend_elements, loc='center left', 
                      bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)
    for text in legend.get_texts():
        if text.get_text().endswith(':'):
            text.set_weight('bold')
            text.set_fontsize(10)
    return legend

def get_metrics(res, metric, regression=False):
    """Extract metrics based on error type."""
    if regression:
        metric_map = {
            'MSE': (res[0], res[3]),
            'RMSE': (np.sqrt(res[0]), np.sqrt(res[3])),
            'MAE': (res[1], res[4]),
            'R2': (res[2], 1 - (res[3]/(res[5] + 0.00001) ))
        }
    else: 
        metric_map = {
            'Accuracy': (res[0], res[3]),
        }
    return metric_map.get(metric, (None, None))

def calculate_diff(metrics_gx, metrics_const, metric,):
    """Calculate performance difference with direction handling for R2."""
    diff = metrics_const - metrics_gx
    if metric in ["R2", "Acciracy", "F1", "Precision", "Recall", "AUC"]:
        diff = -diff
    return diff

def get_filtered_diff(mean_diffs, filter):
    """Apply filter to mean differences."""
    filters = {
        'min': np.min,
        'max': np.max,
        'median': np.median,
        'mean': np.mean
    }
    return filters.get(filter, lambda x: x[filter])(mean_diffs) if isinstance(filter, str) else mean_diffs[filter]
def get_knn_vs_metric_data(res_model, model_name, mapping, filter, metric, compute_difference, distance, regression=False, random_seed=42):
    """Extract kNN and performance difference data."""
    if regression:
        from src.utils.process_results import load_results_regression as load_results
        from src.utils.process_results import get_best_metrics_of_knn_regression as get_best_metrics
    else:
        from src.utils.process_results import load_results_clf as load_results
        from src.utils.process_results import get_best_metrics_of_knn_clf as get_best_metrics
    results = []
    for dataset, files in res_model.items():
        try:
            kw_files = get_kernel_widths_to_filepaths(files)
            if not kw_files: continue
            
            res, _ = load_results(kw_files[int(np.median(range(len(kw_files))))][1])
            metrics_gx, metrics_const = get_metrics(res, metric, regression)
            if metrics_gx is None: continue
            if compute_difference:
                diffs = calculate_diff(metrics_gx, metrics_const, metric)
                mean_metric = np.mean(diffs, axis=1)
            else: 
                mean_metric = np.mean(metrics_gx, axis=1)
            filtered_diff = get_filtered_diff(mean_metric, filter)
            synthetic = "syn" in dataset
            dataset_name = dataset if not synthetic else mapping.get(dataset, dataset)
            knn_result = get_best_metrics(model_name, dataset_name, f"{metric} $g_x$", synthetic=synthetic, distance_measure=distance, random_seed=random_seed)
            if knn_result:
                results.append((dataset, knn_result[0], filtered_diff))
        except Exception as e:
            print(f"Error processing {dataset} for {model_name}: {str(e)}")
    return results

def plot_knn_metrics_vs_metric(models, 
                               method, 
                               datasets, 
                               distance="euclidean",
                               ax=None,
                               marker_size=80, 
                               filter="max", 
                               metric="MSE", 
                               compute_difference = False, 
                               regression=False, 
                               random_seed=42):
    """Main plotting function comparing model complexity vs performance difference."""
    ax = ax or plt.subplots(figsize=(9, 7))[1]
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    cmap = plt.cm.tab20
    
    # Collect and organize all data
    all_results = defaultdict(list)
    for model in models:
        model_results = get_results_files_dict(method, [model], datasets, distance, random_seed=random_seed)[model]
        mapping = get_synthetic_dataset_mapping(datasets, regression)
        points = get_knn_vs_metric_data(model_results, model, mapping, filter, metric, compute_difference, distance, regression)
        for dataset, metric_res, diff in points:
            if "Accuracy" in metric:
                metric_res = (1 - metric_res)*2
            all_results[model].append((dataset, metric_res, diff))
    
    unique_datasets = sorted({d for m in all_results for d, _, _ in all_results[m]}, key=extract_sort_keys)
    colors = {d: cmap(i % 20) for i, d in enumerate(unique_datasets)}
    
    for i, model in enumerate(models):
        marker = markers[i % len(markers)]
        for dataset, metric_res, diff in all_results.get(model, []):
            ax.scatter(np.round(metric_res, 2), diff, color=colors[dataset], marker=marker, 
                      s=marker_size, alpha=0.8)
    if compute_difference:
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
    elif "Accuracy" in metric:
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.7)

    complexity_label = "(1-accuracy)*2" if "Accuracy" in metric else metric
    ax.set_xlabel(f'Complexity $f(x)$ ({complexity_label} of kNN-{"regressor" if regression else "classifier"})')
    method_title = ' '.join(method.split('/')[-1].split('_'))
    if compute_difference:
        if "Accuracy" in metric:
            filter_label = (f'{filter}_k\\{{\\bar{{A}}_g(f,k;m) - \\bar{{A}}_t(f,k;m)\\}}_k' if isinstance(filter, str) 
                    else f'\\bar{{A}}_g(f,k;m) - \\bar{{A}}_t(f,k;m)')
        else:
            filter_label = (f'{filter}_k\\{{\\bar{{{metric}}}_g(f,k;m) - \\bar{{{metric}}}_t(f,k;m)\\}}_k' if isinstance(filter, str) 
                    else f'\\bar{{{metric}}}_g(f,k;m) - \\bar{{{metric}}}_t(f,k;m)')
    else:
        filter_label = metric
        
    ax.set_ylabel(f'${filter_label}$')
    ax.set_title(f"{method_title.capitalize()}: Complexity vs {filter} {metric} {'improvement' if compute_difference else 'of $g_x$'}")
    ax.grid(True, alpha=0.3)
    
    # Adjust layout and add legend
    ax.set_position([ax.get_position().x0, ax.get_position().y0, 
                    ax.get_position().width * 0.8, ax.get_position().height])
    create_dual_legend(ax, unique_datasets, colors, models, markers)
    plt.tight_layout()
    return ax


