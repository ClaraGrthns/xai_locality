import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import gridspec, lines
from collections import defaultdict
import time
import colorcet as cc
from src.utils.process_results import get_knn_vs_diff_model_performance

from src.utils.process_results import get_results_files_dict, get_kernel_widths_to_filepaths, get_synthetic_dataset_mapping
# Set global matplotlib style for all plotting functions
plt.style.use('seaborn-v0_8-ticks')
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.axisbelow'] = True

# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['axes.facecolor'] = 'white'
# plt.rcParams['axes.grid'] = True
# plt.rcParams['axes.grid.which'] = 'major'
# plt.rcParams['axes.grid.axis'] = 'both'
# plt.rcParams['grid.alpha'] = 0.3
# plt.rcParams['grid.linestyle'] = ':'
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.labelsize'] = 12
# plt.rcParams['axes.titlesize'] = 14
# plt.rcParams['xtick.labelsize'] = 10
# plt.rcParams['ytick.labelsize'] = 10
# plt.rcParams['legend.fontsize'] = 10
MARKERS = ['o', 's', '^', 'D', 'v', '<', 'p', '*','>'  ]
MODELS = [
"LogReg",
"MLP",
"LightGBM",
"TabTransformer",
"TabNet",
"ResNet",
"FTTransformer",
]
MODEL_TO_MARKER = {model: marker for model, marker in zip(MODELS, MARKERS)}
COLORMAP = cc.glasbey_dark
datasets_clf = [
    "diabetes130us",
    "credit",
    "jannis",
    "higgs",
    "MiniBooNE",
    "california",
    "bank_marketing",
    "magic_telescope",
    "house_16H",
]
categorical_datasets_clf = [
    "mushroom",
    "albert",
    "road_safety",
    "kdd_census_income",
    "electricity",
    "adult_census_income",
    "adult",
]
real_world_clf = list(set(datasets_clf + categorical_datasets_clf))

CLF_DATASETS = real_world_clf + ['syn \n(d:50, inf f.:2, clust.:2, sep.:0.9, hc: ✓)',
 'syn \n(d:50, inf f.:10, clust.:3, sep.:0.9, hc: ✓)',
 'syn \n(d:100, inf f.:50, clust.:3, sep.:0.9, hc: ✓)',
 'syn \n(d:40, inf f.:20, clust.:10, sep.:0.7, hc: ×)',
 'syn \n(d:25, inf f.:5, clust.:4, sep.:0.8, hc: ×)',
 'syn \n(d:25, inf f.:10, clust.:4, sep.:0.8, hc: ✓)',
 'syn \n(d:55, inf f.:30, clust.:10, sep.:0.6, hc: ×)',
 'syn \n(d:55, inf f.:30, clust.:5, sep.:0.8, hc: ×)',
 'syn \n(d:55, inf f.:30, clust.:10, sep.:0.7, hc: ✓)',
 'syn \n(d:40, inf f.:20, clust.:10, sep.:0.7, hc: ✓)',
 'syn \n(d:55, inf f.:20, clust.:20, sep.:0.3, hc: ×)',
 'syn \n(d:100, inf f.:60, clust.:50, sep.:0.3, hc: ×)',
 'syn \n(d:100, inf f.:10, clust.:50, sep.:0.8, hc: ×)',
 'syn \n(d:100, inf f.:60, clust.:20, sep.:0.2, hc: ×)',
 'syn \n(d:200, inf f.:20, clust.:50, sep.:0.5, hc: ×)',
 'syn \n(d:50, inf f.:10, clust.:10, sep.:0.2, hc: ×)',
 'syn \n(d:50, inf f.:10, clust.:40, sep.:0.2, hc: ×)']



real_world_reg= list(set(
    ["california_housing" ,
    "diamonds",
    "elevators" ,
    "medical_charges" ,
    "superconduct" ,
    "houses",
    "allstate_claims_severity",
    "sgemm_gpu_kernel_performance",
    "diamonds",
    "particulate_matter_ukair_2017",
    "seattlecrime6",
    "airlines_DepDelay_1M",
    "delays_zurich_transport",
    "nyc-taxi-green-dec-2016",
    "microsoft" ,
    "year"]
    ))

REG_DATASETS = real_world_reg + [ 'syn linear \n(d:30, inf f.:20, noise:0.6)',
 'syn piecewise_linear \n(d:15, inf f.:10, noise:0.2)',
 'syn piecewise \n(d:60, inf f.:15, noise:0.25)',
 'syn polynomial \n(d:50, inf f.:25, noise:0.8)',
 'syn adv_polynomial \n(d:80, inf f.:40, noise:0.5)',
 'syn hierarchical \n(d:70, inf f.:25, noise:0.15)',
 'syn poly_interaction \n(d:90, inf f.:40, noise:0.1)',
 'syn exponential_interaction \n(d:50, inf f.:10, noise:0.2)',
 'syn polynomial \n(d:100, inf f.:10, noise:0.0)',
 'syn polynomial \n(d:100, inf f.:50, noise:0.0)',
 'syn interaction \n(d:50, inf f.:30, noise:0.1)',
 'syn poly_interaction \n(d:90, inf f.:70, noise:0.1)',
 'syn multiplicative_chain \n(d:70, inf f.:30, noise:0.5)',
 'syn sigmoid_mix \n(d:200, inf f.:80, noise:0.15)',
 'syn exponential_interaction \n(d:20, inf f.:10, noise:0.2)',
 'syn polynomial \n(d:20, inf f.:10, noise:0.4)',
 'syn polynomial \n(d:20, inf f.:5, noise:0.1)']

COLOR_TO_REG_DATASET = {
    dataset: color for dataset, color in zip(REG_DATASETS, COLORMAP)
}
COLOR_TO_CLF_DATASET = {
    dataset: color for dataset, color in zip(CLF_DATASETS, COLORMAP)
}
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
    "MAE const. local model - MAE $g_x$": (4, 1),
    "MSE $g_x$ / MSE const. local model": (0, 3),
    "MSE $g_x$ / Variance $f(x)$": (0, 5),
    "E(MSE $g_x$ / Variance $f(x)$)": (0, 5),
    "MSE const. local model / Variance $f(x)$": (3, 5),
}
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
LINESTYLES = [  "--","-", "-."]
KERNEL_LABELS = ["default/2", "default", "default*2"]

def plot_metric(ax, values, neighbors, color, style, max_neighbors=None):
    """Plot a single metric line."""
    if values is None:
        print("Warning: Missing values")
        return
    neighbors = neighbors[:max_neighbors]
    values = values[:max_neighbors]
    ax.plot(neighbors, values, color=color, 
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

def create_legend(models, colors, method, unique_kw_lines_idx=[]):
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

def get_default_kernel_width_path(files):
    kernel_widths_fp = get_kernel_widths_to_filepaths(files)
    file_paths = [path for width, path in kernel_widths_fp]
    kernel_widths = np.array([kw[0] for kw in kernel_widths_fp])
    default_kw = np.median(kernel_widths)
    return file_paths[np.where(kernel_widths == default_kw)[0][0]]

def get_fraction(metr0, metr1):
    metr1 = np.round(metr1, decimals=5)
    metr1 = np.where(np.isclose(metr1, 0), np.nan, metr1)  # Avoid division by zero
    vals  = np.where(np.isclose(metr0, 0), 0, metr0 / metr1) # if g_x = 0, then value as zero mistakes
    return vals

def plot_dataset_metrics(models, 
                         datasets, 
                         method, 
                         metrics, 
                         distance="euclidean", 
                        max_neighbors=None, 
                        save=False, 
                        lime_features=10, 
                        regression=False, 
                        scale_by_variance = False, 
                        random_seed=False, 
                        summarizing_statistics=None):
    """Main plotting function."""
    plt.style.use('seaborn-v0_8-ticks')
    if regression:
        from src.utils.process_results import load_results_regression as load_results
        metrics_map = METRICS_MAP_REG
    else:
        from src.utils.process_results import load_results_clf as load_results
        metrics_map = METRICS_TO_IDX_CLF
    results = get_results_files_dict(method, models, datasets, distance, lime_features, random_seed=random_seed)
    colors = {m: plt.cm.tab10(i) for i, m in enumerate(models)}
    if summarizing_statistics is None:
        summarizing_statistics = lambda x, axis: np.mean(x, axis=axis)
    for dataset in set(d for model_data in results.values() for d in model_data):
        fig, axes, legend_ax = setup_plot(len(metrics))
        dataset_models = [m for m in results if dataset in results[m]]
        for ax_idx, (ax, metric) in enumerate(zip(axes, metrics)):
            metric_idx = metrics_map[metric]
            is_diff = "-" in metric
            is_ratio = "/" in metric
            for model in dataset_models:
                files = results[model][dataset]
                if isinstance(files, list) and len(files) == 0:
                    continue
                if method == "lime" and metric not in ["Variance $f(x)$", "Radius", "Accuracy const. local model", "MSE const. local model", "MAE const. local model"]:
                    path = get_default_kernel_width_path(files)
                else:
                    path = files[0] if isinstance(files, list) else files
                # print(model, dataset, path)
                data, neighbors = load_results(path)
                neighbors = np.arange(0, len(neighbors))
                neighbors = neighbors + 1 if neighbors[0] == 0 else neighbors
                if metric == "E(MSE $g_x$ / Variance $f(x)$)":
                    metr0 = data[metric_idx[0]]
                    metr1 = data[metric_idx[1]]
                    summary_0 = summarizing_statistics(metr0, axis=1)
                    summary_1 = summarizing_statistics(metr1, axis=1)
                    summary_vals = get_fraction(summary_0, summary_1)
                else:
                    if is_ratio:
                        metr0 = data[metric_idx[0]]
                        metr1 = data[metric_idx[1]]
                        vals = get_fraction(metr0, metr1)# if g_x = 0, then value as zero mistakes
                    elif is_diff:
                        vals = data[metric_idx[0]] - data[metric_idx[1]]
                        if scale_by_variance and regression:
                            vals /= data[metrics_map["Variance $f(x)$"]]
                    else:
                        vals = data[metric_idx]
                    summary_vals = summarizing_statistics(vals, axis=1)

                plot_metric(ax, summary_vals, neighbors, colors[model], '-', max_neighbors)
            if is_ratio:
                ax.axhline(1, color='k', alpha=0.8, linewidth=0.8)
            elif is_diff:
                ax.axhline(0, color='k', alpha=0.8, linewidth=0.8)
                
            if is_diff and (not is_ratio) and scale_by_variance:
                ax.set_title(f"({metric})$\\times \\frac{{1}}{{\\text{{var}}(f(x'))}}$")
            else:
                ax.set_title(metric)
            ax.set_xlabel(f"Neighborhood size ({distance} distance)")
            ax.grid(True, linestyle=':')
        handles, labels = create_legend(dataset_models, colors, method, [])
        legend_ax.legend(handles, labels, frameon=True, fontsize=9)
        method_title = method.split("/")[-1]
        method_title = " ".join(method_title.split("_"))
        if method == "lime" and lime_features == "all":
            method_title = method_title + " (all features)"
        title = f"{method_title.capitalize()} on {dataset.capitalize()}"
        if method == "lime" and lime_features == "all":
            title += " (all features)"
        y_position = 1.04 if "syn" in dataset else 1.02
        fig.suptitle(title, fontsize=14, y=y_position)
        if save:
            fig.savefig(
                f"graphics/knn_vs_metrics_{method.split('/')[-1]}_{dataset}.pdf",
                bbox_inches='tight',  # Only if needed
                dpi=150,              # Reduced from 300
                # optimize=True,         # Enable PDF optimizations
                metadata={'CreationDate': None}  # Disable timestamp
            )
        else:
            plt.show();


def extract_sort_keys(dataset, regression = False):
    """Extract sorting keys from dataset name using regex."""
    if regression: 
        # Try to match polynomial regression pattern like: syn-reg polynomial (d:100, if:10, b: 1.0, ns:0.0, er:60)
        match = re.search(r'syn\s+\w+\s+\(d:(\d+),\s*inf f.:(\d+)', dataset)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (int(match.group(1)), int(match.group(2))) if match else (0., 0.)
    else:
        pattern = r'd:(\d+).*?inf f.:(\d+).*?clust.:(\d+).*?sep.:([\d\.]+)' # inf f.:{i}, clust.:{c}, sep.:
        match = re.search(pattern, dataset)
        if match:
            n_feat = int(match.group(1))
            inf_feat = int(match.group(2))
            n_clusters = int(match.group(3))
            class_sep = float(match.group(4))
            return n_feat, inf_feat,  n_clusters, -class_sep # Negative class_sep for descending order
        return 0, 0, 0, 0  # Default for non-matching datasets


def create_dual_legend(ax, datasets, colors, models, markers, bbox_to_anchor=(1.02, 0.5), plot_multiple=False):
    """Create a side-by-side legend with datasets on the left and models on the right."""
    if plot_multiple:
        ax.axis("off")
    dataset_handles = [
        plt.Line2D([], [], marker='o', color=colors.get(d, "black"), label=d, linestyle='None')
        for d in datasets
    ]
    model_handles = [
        plt.Line2D([], [], marker=markers[m], color='gray', label=m, linestyle='None')
        for i, m in enumerate(models)
    ]
    handles = (
        [plt.Line2D([], [], color='none', label='Datasets:')] + dataset_handles +
        [plt.Line2D([], [], color='none', label='Models:')] + model_handles
    )
    legend = ax.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=bbox_to_anchor, #(0.0, 0.5),
        frameon=False,
        fontsize=9,
        ncol=2,
        handletextpad=1.5,
        columnspacing=2.0,
        borderpad=1.0
    )
    for text in legend.get_texts():
        if text.get_text() in ["Datasets:", "Models:"]:
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
            'R2': (res[2], 1 - (res[3]/(res[5] + 0.00001) )), 
            'Variance': res[5]
        }
    else: 
        metric_map = {
            'Accuracy': (res[0], res[3]),
        }
    return metric_map.get(metric, (None, None))

def calculate_diff(metrics_gx, metrics_const, metric, regression):
    """Calculate performance difference with direction handling for R2."""
    diff = metrics_gx - metrics_const
    if regression:
        diff = metrics_const - metrics_gx
    return diff

def get_filter(mean_diffs, filter):
    """Apply filter to mean differences."""
    filters = {
        'min': np.nanmin,
        'max': np.nanmax,
        'median': np.nanmedian,
        'mean': np.nanmean
    }
    return filters.get(filter, lambda x: x[filter])(mean_diffs) if isinstance(filter, str) else mean_diffs[filter]

def get_fraction(metr0, metr1):
    metr1 = np.where(np.isclose(metr1, 0), np.nan, metr1)  # Avoid division by zero
    return 1 - metr0/metr1

def edit_ticks(ticks, val, label):
    ticks = [t for t in ticks if t < 1.1]  # filter out > 1.1
    if not any(np.isclose(t, val) for t in ticks):
        ticks.append(val)
    ticks = sorted(set(ticks))
    ticks_labels = [label if np.isclose(t, val) else str(round(t, 2)) for t in ticks]
    return ticks, ticks_labels

def get_knn_vs_metric_data(res_model, 
                           model_name, 
                           mapping, 
                           filter, 
                           metric, 
                           distance, 
                           regression=False, 
                           random_seed=42, 
                           difference_to_constant_model=False,
                           summarizing_statistics=None,
                           complexity_regression=False, 
                           average_over_n_neighbors=200):
    """Extract kNN and performance difference data."""
    if regression:
        from src.utils.process_results import load_results_regression as load_results
        from src.utils.process_results import get_best_metrics_of_complexity_of_f_regression as get_best_metrics
        metrics_map = METRICS_MAP_REG
    else:
        from src.utils.process_results import load_results_clf as load_results
        from src.utils.process_results import get_best_metrics_of_complexity_of_f_clf as get_best_metrics
        metrics_map = METRICS_TO_IDX_CLF
    results = []
    if summarizing_statistics is None:
        summarizing_statistics = lambda x, axis: np.nanmedian(x, axis=axis)
    for dataset, files in res_model.items():
        kw_files = get_kernel_widths_to_filepaths(files)
        if len(kw_files)==0: continue
        data, _ = load_results(kw_files[int(np.median(range(len(kw_files))))][1])
        metric_idx = metrics_map[metric]
        is_diff = "-" in metric
        is_ratio = "/" in metric
        if is_ratio:
            vals = get_fraction(data[metric_idx[0]], data[metric_idx[1]])
            summary_vals = summarizing_statistics(vals, axis=1)[:average_over_n_neighbors]
            if difference_to_constant_model:
                metric_constant = metric.replace("$g_x$", "const. local model")
                metric_idx_constant = metrics_map[metric_constant]
                vals_constant = get_fraction(data[metric_idx_constant[0]], data[metric_idx_constant[1]])
                summary_vals_constant = summarizing_statistics(vals_constant, axis=1)[:average_over_n_neighbors]
                summary_vals = summary_vals - summary_vals_constant
        elif is_diff:
            vals = data[metric_idx[0]] - data[metric_idx[1]]
            summary_vals = summarizing_statistics(vals, axis=1)[:average_over_n_neighbors]
        else:
            vals = data[metric_idx]
            summary_vals = summarizing_statistics(vals, axis=1)[:average_over_n_neighbors]
        filtered_res = get_filter(summary_vals, filter)
        synthetic = "syn" in dataset
        dataset_name = dataset if not synthetic else mapping.get(dataset, dataset)
        if regression:
            if complexity_regression:
                complexity_metrics = "R2 Lin Reg"
            else:
                complexity_metrics = "R2 $g_x$"
        else:
            if complexity_regression:
                complexity_metrics = "Accuracy Log Reg"
            else:
                complexity_metrics = "Accuracy $g_x$"
        res_complexity = get_best_metrics(model_name, 
                                            dataset_name, 
                                            complexity_metrics, 
                                            synthetic=synthetic, 
                                            distance_measure=distance,
                                            complexity_regression=complexity_regression,
                                            random_seed=random_seed)
        if res_complexity is None: continue
        compl_metr = np.min(res_complexity[0], 0)
        knn_result_metric = compl_metr
        results.append((dataset,knn_result_metric, filtered_res))
    return results

def get_y_axis_label(filter, metric_axis_label, is_diff, is_ratio, summary):
    if summary is np.nanmean:
        summary_label = "\\mathbb{E}_x"
    else:
        summary_label = "Median_x"
    if is_diff:
        if "Accuracy" in metric_axis_label:
            label = (
                f"$\\text{{{filter}}}_k\\{{{summary_label} [{{A}}_g(f, k; m) - {summary_label}{{A}}_t(f, k; m)]\\}}$"
                if isinstance(filter, str)
                else f"{summary_label} [${{A}}_g(f, {{{filter}}}; m) - {summary_label}{{A}}_t(f,{{{filter}}};m))$"
            )
        else:
            label = (
                f"$\\text{{{filter}}}_k \\{{{summary_label} [{{{metric_axis_label}}}_t(f, k; m) - {summary_label}{{{metric_axis_label}}}_g(f, k; m)]\\}}$"
                if isinstance(filter, str)
                else f"{summary_label} [${{{metric_axis_label}}}_t(f, {filter}; m) - {summary_label}{{{metric_axis_label}}}_g(f, {filter}; m)) $"
            )
    elif is_ratio:
        # label = (
        #     f"$\\text{{{filter}}}_k \\{{{summary_label} \\frac{{{{{metric_axis_label}}}_g(f, k; m)}}{{Var(f, k; m)}}\\}}$"
        #     if isinstance(filter, str)
        #     else f"${summary_label} \\frac{{{{{metric_axis_label}}}_g(f, {filter}; m)}}{{Var(f, {filter}; m)}} $"
        # )
        label = (
            f"$\\text{{{filter}}}_k \\{{{summary_label} [ \\text{{R}}^2(g_x, f, k; m) ]\\}} $"
            if isinstance(filter, str)
            else f"$\\{{{summary_label}[\\text{{R}}^2(g_x, f, {{{filter}}}; m)}}]$"
        )
    else:
        label = (f"${summary_label}$ [{metric_axis_label}](f, {filter}; m)" 
        if isinstance(filter, int) 
        else f"$\\text{{{filter}}}_k \\{{{summary_label} [ \\text{{Accuracy}}(g_x, f, k; m) ]\\}} $")
    return label

def plot_knn_metrics_vs_metric(models, 
                               method, 
                               datasets, 
                               distance="euclidean",
                               ax=None,
                               filter="max", 
                               metric="MSE $g_x$ / Variance $f(x)$", 
                               difference_to_constant_model=False,
                               regression=False, 
                               summarizing_statistics=None,
                               random_seed=42,
                               complexity_regression = False,
                               average_over_n_neighbors=200,
                               save=False):
    """Main plotting function comparing model complexity vs performance difference."""
    
    create_legend_to_ax = ax is None
    if ax is None:
        ax = plt.subplots(figsize=(9, 7))[1]
    cmap = COLOR_TO_CLF_DATASET if not regression else COLOR_TO_REG_DATASET
    res_dict = get_results_files_dict(method, models, datasets, distance, random_seed=random_seed)
    is_diff = "-" in metric
    is_ratio = "/" in metric    
    all_results = defaultdict(list)
    cutoff_at = 0 if (regression and is_ratio) else 0.5
    cutoff_value_replaced = -0.05 if regression and is_ratio else 0.45
    cutoff_label = f"<{cutoff_at}"
    if summarizing_statistics is None:
        summarizing_statistics = lambda x, axis: np.nanmean(x, axis=axis)
    for model in models:
        model_results = res_dict.get(model, None)
        if model_results is None or len(model_results) == 0:
            print(f"No results found for {model}.")
            continue
        mapping = get_synthetic_dataset_mapping(datasets, regression)
        points = get_knn_vs_metric_data(model_results, 
                                        model, 
                                        mapping, 
                                        filter, 
                                        metric, 
                                        distance, 
                                        regression, 
                                        summarizing_statistics=summarizing_statistics,
                                        average_over_n_neighbors=average_over_n_neighbors,
                                        difference_to_constant_model=difference_to_constant_model,
                                        complexity_regression=complexity_regression,)
        for dataset, knn_metric_res, filtered_res in points:
            all_results[model].append((dataset, knn_metric_res, filtered_res))

    unique_datasets = sorted({d for m in all_results for d, _, _ in all_results[m]}, key=lambda x: extract_sort_keys(x, regression))
    colors = {d: cmap[d] for i, d in enumerate(unique_datasets)}
    models_plotted = list(all_results.keys())
    markers_to_models = {m: MODEL_TO_MARKER[m] for i, m in enumerate(models_plotted)}
    current_min_x_y = 1
    for i, model in enumerate(models_plotted):
        x_vals, y_vals, colors_list = [], [], []
        for dataset, knn_metric_res, filtered_res in all_results[model]:
            x_vals.append(np.round(knn_metric_res, 4))
            y_vals.append(filtered_res if filtered_res > cutoff_at else cutoff_value_replaced)
            colors_list.append(colors[dataset])
        current_min_x_y = min(current_min_x_y, min(x_vals), min(y_vals))
        ax.scatter(x_vals, y_vals, c=colors_list, marker=markers_to_models[model], 
                   s=80, alpha=0.8)

    # === Handle axes ===
    if is_diff or "R2" in metric:
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
    elif is_ratio:
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
        ax.set_ylim((0, 1.1))
        yticks = ax.get_yticks()
        yticks, ytick_labels = edit_ticks(yticks, cutoff_value_replaced, cutoff_label)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        ax.set_xlim((0, 1))
    elif "Accuracy" in metric:
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
        current_min_x_y = np.max([cutoff_value_replaced, current_min_x_y])
        current_min_x_y = np.min([0.5, current_min_x_y])
        ax.plot([current_min_x_y, 1], [current_min_x_y, 1], linestyle='--', color='gray', alpha=0.7)
        yticks = list(ax.get_yticks())
        yticks, ytick_labels = edit_ticks(yticks, cutoff_value_replaced, cutoff_label)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
    # === Labels ===
    metric_axis_label = "$R^2$" if regression else "Accuracy"
    if complexity_regression:
        x_label = f'{metric_axis_label} of {'Linear' if regression else 'Logistic'} Regression on model predictions'
    else: 
        x_label = f'{metric_axis_label} of kNN-{"regressor" if regression else "classifier"} on model predictions'
    ax.set_xlabel(x_label)
    method_title = ' '.join(method.split('/')[-1].split('_'))
    y_axis_label = get_y_axis_label(filter, metric, is_diff, is_ratio, summary=summarizing_statistics)
    ax.set_ylabel(y_axis_label)
    ax.set_xlim((0, 1)) if regression else  ax.set_xlim((0.5, 1))

    # === Title ===
    if isinstance(filter, int):
        ax.set_title(f"{method_title.capitalize()}:\n Complexity of f vs. {metric_axis_label} average {'improvement' if is_diff else 'of $g_x$'} within {filter} neighbors")
    else:
        ax.set_title(f"{method_title.capitalize()}:\n Complexity of f vs. {filter} average {metric_axis_label} {'improvement' if is_diff else 'of $g_x$'}")

    ax.grid(True, alpha=0.3)
    ax.set_position([
        ax.get_position().x0,
        ax.get_position().y0,
        ax.get_position().width * 0.8,
        ax.get_position().height
    ])
    if create_legend_to_ax:
        create_dual_legend(ax, unique_datasets, colors, models_plotted, markers_to_models, bbox_to_anchor=(1.02, 0.5))

    if save:
        plt.savefig(
            f"graphics/complexity_vs_{filter}_metrics_{method.split('/')[-1]}_{dataset}.pdf",
            bbox_inches='tight',
            dpi=100,
            metadata={'CreationDate': None}
        )
    return ax, unique_datasets, colors, models_plotted, list(markers_to_models.values())


def extract_random_seed(file_path):
    match = re.search(r'random_seed-(\d+)', file_path)
    return float(match.group(1)) if match else float('inf')
def order_by_seed(file_paths):
    """Order file paths by the downsampling fraction in the string."""
    return sorted(file_paths, key=extract_random_seed)

def plot_random_seeds_results_per_dataset(method, models, datasets, regression=False):
    """
    Plots the model explanation accuracy vs. neighborhood size for given models and datasets.

    Parameters:
        method (str): The method to use ('lime' or 'gradient_methods/integrated_gradient').
        models (list): List of models to evaluate.
        datasets (list): List of datasets to evaluate.
        regression (bool): Whether the task is regression (True) or classification (False).
    """
    if regression:
        from src.utils.process_results import load_results_regression as load_results
    else:
        from src.utils.process_results import load_results_clf as load_results
        
    colors = {m: plt.cm.tab10(i) for i, m in enumerate(models)}
    alpha = 0.2  

    results_dict = get_results_files_dict(method, models, datasets, downsampled=False, random_seed=True)

    for dataset_idx, dataset in enumerate(datasets):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.tick_params(axis='both', which='both', length=5)
        for model_idx, model in enumerate(models):
            res_fp_ls = results_dict[model].get(dataset, None)
            if res_fp_ls is None:
                continue
            res_fp_ls = order_by_seed(res_fp_ls)
            mean_metric_ls = []
            for i in range(len(res_fp_ls)):
                res, knn = load_results(res_fp_ls[i])
                metric_red = res[0]
                mean_metric = np.mean(metric_red, axis=1)
                if len(mean_metric[:200]) != 200:
                    print(f"Warning: {model} {dataset} has {len(mean_metric[:200])} values instead of 200.")
                    continue
                mean_metric_ls.append(mean_metric[:200])
            mean_metric_ls = np.array(mean_metric_ls)
            mean_values = np.mean(mean_metric_ls, axis=0)
            std_values = np.std(mean_metric_ls, axis=0)
            
            ax.plot(knn, mean_values, 
                    label=model,
                    color=colors[model],
                    linewidth=2)
            ax.fill_between(knn,
                           mean_values - std_values,
                           mean_values + std_values,
                           color=colors[model],
                           alpha=alpha)
        ax.set_xlabel("Number of Nearest Neighbors (k)", fontsize=12)
        ax.set_ylabel(f"{'MSE' if regression else 'Accuracy'} of $g_x$", fontsize=12)
        ax.set_title(f"{method.split("/")[-1].capitalize()}: Averaged over 5 random seeds, {'MSE' if regression else 'Accuracy'} vs. Neighborhood Size\nDataset: {dataset}", 
                    fontsize=14, pad=20)
        ax.grid(True, linestyle='--', alpha=0.6)
        # ax.set_axisbelow(True)
        legend = ax.legend(title="Models", 
                          fontsize=10,
                          title_fontsize=11,
                          frameon=True,
                          framealpha=0.9,
                          loc='lower right')
        plt.tight_layout()
        plt.show();

def get_metric_vals(is_ratio, is_diff, metric_idx, data):
    if is_ratio:
        vals = get_fraction(data[metric_idx[0]], data[metric_idx[1]])
    elif is_diff:
        vals = data[metric_idx[0]] - data[metric_idx[1]]
    else:
        vals = data[metric_idx]
    return vals
        
def get_local_vs_constant_metric_data(res_model, 
                           filter, 
                           metric, 
                           regression=False, 
                           summarizing_statistics=None,
                           average_over_n_neighbors=200):
    """Extract kNN and performance difference data."""
    if regression:
        from src.utils.process_results import load_results_regression as load_results
        metrics_map = METRICS_MAP_REG
    else:
        from src.utils.process_results import load_results_clf as load_results
        metrics_map = METRICS_TO_IDX_CLF
    results = []
    if summarizing_statistics is None:
        summarizing_statistics = lambda x, axis: np.nanmedian(x, axis=axis)
    for dataset, files in res_model.items():
        kw_files = get_kernel_widths_to_filepaths(files)
        if len(kw_files)==0: continue
        data, _ = load_results(kw_files[int(np.median(range(len(kw_files))))][1])
        # process local xai model results =========
        metric_idx = metrics_map[metric]
        is_diff = "-" in metric
        is_ratio = "/" in metric
        vals_g = get_metric_vals(is_ratio, is_diff, metric_idx, data)
        summary_vals_g = summarizing_statistics(vals_g, axis=1)[:average_over_n_neighbors]
        filtered_res_g = get_filter(summary_vals_g, filter)

        # process constant xai model results =========
        metric_constant = metric.replace("$g_x$", "const. local model")
        metric_idx_constant = metrics_map[metric_constant]
        vals_constant = get_metric_vals(is_ratio, is_diff, metric_idx_constant, data)
        summary_vals_constant = summarizing_statistics(vals_constant, axis=1)[:average_over_n_neighbors]
        filtered_res_constant = get_filter(summary_vals_constant, filter)
        results.append((dataset, filtered_res_constant, filtered_res_g))
    return results

    
def plot_local_metrics_vs_constant_metric(models, 
                               method, 
                               datasets, 
                               distance="euclidean",
                               ax=None,
                               filter="max", 
                               metric="MSE $g_x$ / Variance $f(x)$", 
                               regression=False, 
                               summarizing_statistics=None,
                               average_over_n_neighbors=200,
                               save=False):
    """Main plotting function comparing model complexity vs performance difference."""
    
    create_legend_to_ax = ax is None
    if ax is None:
        ax = plt.subplots(figsize=(9, 7))[1]
    cmap = COLOR_TO_REG_DATASET if regression else COLOR_TO_CLF_DATASET
    res_dict = get_results_files_dict(method, models, datasets, distance, random_seed=False)
    is_diff = "-" in metric
    is_ratio = "/" in metric    
    cut_off = 0 if regression else 0.5
    replace_with = -0.05 if regression else 0.4
    all_results = defaultdict(list)

    if summarizing_statistics is None:
        summarizing_statistics = lambda x, axis: np.nanmean(x, axis=axis)

    for model in models:
        model_results = res_dict.get(model, None)
        if model_results is None or len(model_results) == 0:
            print(f"No results found for {model}.")
            continue
        points = get_local_vs_constant_metric_data(res_model = model_results,
                           filter=filter,
                            metric=metric,
                            regression=regression, 
                            summarizing_statistics=summarizing_statistics,
                            average_over_n_neighbors=average_over_n_neighbors)
        for dataset, metr_constant, metr_g in points:
            all_results[model].append((dataset, metr_constant, metr_g))

    unique_datasets = sorted({d for m in all_results for d, _, _ in all_results[m]}, key=lambda x: extract_sort_keys(x, regression))
    colors = {d: cmap[d] for i, d in enumerate(unique_datasets)}
    models_plotted = list(all_results.keys())
    markers_to_models = {m: MODEL_TO_MARKER[m] for i, m in enumerate(models_plotted)}
    current_min_x_y = np.inf
    current_max_x_y = 0
    for i, model in enumerate(models_plotted):
        x_vals, y_vals, colors_list = [], [], []
        for dataset, metr_constant, metr_g in all_results[model]:
            metr_constant = metr_constant if metr_constant >= cut_off else replace_with
            metr_g = metr_g if metr_g >= cut_off else replace_with
            x_vals.append(metr_constant)
            y_vals.append(metr_g)
            colors_list.append(colors[dataset])
        current_min_x_y = min(current_min_x_y, min(x_vals), min(y_vals))
        current_max_x_y = max(current_max_x_y, max(x_vals), max(y_vals))
        ax.scatter(x_vals, y_vals, c=colors_list, marker=markers_to_models[model], 
                   s=80, alpha=0.8)
        
    def edit_ticks(ticks, val, label, exclude_lower = -np.inf, exclude_upper=1.1):
        ticks = [t for t in ticks if exclude_lower<=t<exclude_upper]  # filter out > 1.1
        if not any(np.isclose(t, val) for t in ticks):
            ticks.append(val)
        ticks = sorted(set(ticks))
        ticks_labels = [label if np.isclose(t, val) else str(round(t, 2)) for t in ticks]
        return ticks, ticks_labels
       
    # === Handle axes ===
    if is_diff or "R2" in metric:
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax.set_ylim((replace_with, 1))
        ax.set_xlim((replace_with, 1))
    elif is_ratio:
        ax.plot([current_min_x_y, 1], [current_min_x_y, 1], linestyle='--', color='gray', alpha=0.7)
        ax.set_ylim((current_min_x_y, 1))
        ax.set_xlim((current_min_x_y, 1))
    elif "Accuracy" in metric:
        current_min_x_y = np.max([replace_with, current_min_x_y])
        current_min_x_y = np.min([cut_off, current_min_x_y])
        ax.plot([current_min_x_y, 1], [current_min_x_y, 1], linestyle='--', color='gray', alpha=0.7)
        ax.set_ylim((current_min_x_y, 1))
        ax.set_xlim((current_min_x_y, 1))
    yticks = ax.get_yticks()
    yticks, ytick_labels = edit_ticks(yticks, replace_with, f"<{cut_off}", exclude_lower=current_min_x_y, exclude_upper=1.1)
    ax.set_yticks(sorted(set(yticks)))
    ax.set_yticklabels(ytick_labels)
    xticks = ax.get_xticks()
    xticks, xtick_labels = edit_ticks(xticks, replace_with, f"<{cut_off}", exclude_lower=current_min_x_y-0.1, exclude_upper=1.1)
    ax.set_xticks(sorted(set(xticks)))
    ax.set_xticklabels(xtick_labels)

    # # ax.set_aspect('equal', adjustable='box')
    # # === Labels ===
    metric_axis_label = "MSE/Var" if regression else "Accuracy"
    method_title = ' '.join(method.split('/')[-1].split('_'))
    # y_axis_label = get_y_axis_label(filter, metric, is_diff, is_ratio, summary=summarizing_statistics)
    # x_axis_label = get_y_axis_label(filter, metric.replace("$g_x$", "const. local model"), is_diff, is_ratio, summary=summarizing_statistics)
    y_axis_label = "Best average performance for $g_x$" if isinstance(filter, str) else f"Average performance $g_x$ for {filter} closest neighbors"
    x_axis_label = "Best average performance for const. local model" if isinstance(filter, str) else f"Average performance const. local model for {filter} closest neighbors"
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)

    # === Title ===
    if isinstance(filter, int):
        ax.set_title(f"{method_title.capitalize()}: average {metric_axis_label} of const. local model vs.\n {metric_axis_label} average of $g_x$ within {filter} neighbors")
    else:
        ax.set_title(f"{method_title.capitalize()}: {filter} average {metric_axis_label} of const.\n local model vs. {filter} average {metric_axis_label} of $g_x$")

    ax.grid(True, alpha=0.3)
    ax.set_position([
        ax.get_position().x0,
        ax.get_position().y0,
        ax.get_position().width * 0.8,
        ax.get_position().height
    ])
    if create_legend_to_ax:
        create_dual_legend(ax, unique_datasets, colors, models_plotted, markers_to_models)

    if save:
        plt.savefig(
            f"graphics/complexity_vs_{filter}_metrics_{method.split('/')[-1]}_{dataset}.pdf",
            bbox_inches='tight',
            dpi=100,
            metadata={'CreationDate': None}
        )
    return ax, unique_datasets, colors, models_plotted, list(markers_to_models.values())

def plot_knn_vs_diff_scatter(models, 
                        datasets, 
                        distance="euclidean", 
                        regression=False, 
                        synthetic=False, 
                        random_seed=42,
                        complexity_regression=False, 
                        ax=None, 
                        save=False):
    """
    Scatter plot: x-axis = performance_smple_model_model_preds, y-axis = diff (model - simple model on true labels)
    Each point: (model, dataset). Color by dataset, marker by model.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig = ax.figure

    # Prepare colors and markers
    from src.utils.plotting_utils import COLOR_TO_CLF_DATASET, COLOR_TO_REG_DATASET, extract_sort_keys
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    cmap = COLOR_TO_REG_DATASET if regression else COLOR_TO_CLF_DATASET

    # Gather results
    results = []
    for model in models:
        for dataset in datasets:
            try:
                x, y = get_knn_vs_diff_model_performance(
                    model, dataset, distance=distance, regression=regression,
                    synthetic=synthetic, random_seed=random_seed, complexity_regression=complexity_regression
                )
                results.append((model, dataset, x, y))
            except Exception as e:
                print(f"Skipping {model} on {dataset}: {e}")

    unique_datasets = sorted({d for _, d, _, _ in results}, key=lambda x: extract_sort_keys(x, regression))
    colors = {d: cmap[d] for d in unique_datasets}
    models_plotted = sorted({m for m, _, _, _ in results})
    markers_to_models = {m: markers[i % len(markers)] for i, m in enumerate(models_plotted)}

    for model, dataset, x, y in results:
        ax.scatter(x, y, color=colors[dataset], marker=markers_to_models[model], s=80, alpha=0.8)

    smpl_model = "Lin. Reg" if regression and complexity_regression else ("kNN-reg." if regression else ("Log. Reg" if complexity_regression else "kNN-clf."))
    ax.set_xlabel(f"{smpl_model} on Model Predictions", fontsize=14)
    ax.set_ylabel(f"Difference {"$R^2$" if regression else "Accuracy"}  on true labels: f-{smpl_model}", fontsize=14)
    ax.set_title(f"Simplicty of $f$ vs. Difference: Model - {smpl_model} on true labels", fontsize=16)
    ax.grid(True, alpha=0.3)

    dataset_handles = [
        plt.Line2D([], [], marker='o', color=colors[d], label=d, linestyle='None')
        for d in unique_datasets
    ]
    model_handles = [
        plt.Line2D([], [], marker=markers_to_models[m], color='gray', label=m, linestyle='None')
        for m in models_plotted
    ]
    handles = (
        [plt.Line2D([], [], color='none', label='Datasets:')] + dataset_handles +
        [plt.Line2D([], [], color='none', label='Models:')] + model_handles
    )
    legend = ax.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        ncol=2,
        handletextpad=1.5,
        columnspacing=2.0,
        borderpad=1.0
    )
    for text in legend.get_texts():
        if text.get_text() in ["Datasets:", "Models:"]:
            text.set_weight('bold')
            text.set_fontsize(10)

    plt.tight_layout()
    if save:
        plt.savefig("graphics/knn_vs_diff_scatter.pdf", bbox_inches='tight', dpi=100, metadata={'CreationDate': None})
    plt.show()
    return ax