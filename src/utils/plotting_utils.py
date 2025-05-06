import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import gridspec, lines
from collections import defaultdict
import time
import colorcet as cc
from src.utils.process_results import (get_knn_vs_diff_model_performance,
                                    get_performance_metrics_smpl_complex_models,
                                    get_kw_fp,
                                    get_fraction, 
                                    get_knn_vs_metric_data,
                                    extract_sort_keys,
                                    get_filter,
                                    filter_best_performance_local_model
                                    )
import matplotlib as mpl

font_path = "/home/grotehans/xai_locality/font/cmunbsr.ttf"
mpl.font_manager.fontManager.addfont(font_path)
font_path = "/home/grotehans/xai_locality/font/Computer Modern Roman.ttf"
mpl.font_manager.fontManager.addfont(font_path)
font_path = "/home/grotehans/xai_locality/font/Times New Roman.ttf"
mpl.font_manager.fontManager.addfont(font_path)
from matplotlib.font_manager import FontProperties

from src.utils.process_results import get_results_files_dict, get_kernel_widths_to_filepaths, get_random_seed_to_filepaths, get_downsample_fraction_to_filepaths, get_synthetic_dataset_mapping, get_synthetic_dataset_friendly_name, get_synthetic_dataset_friendly_name_regression
# Set global matplotlib style for all plotting functions
plt.style.use('seaborn-v0_8-ticks')
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.axisbelow': True,
    'axes.grid': True,
    'grid.linestyle': ':',
    'axes.xmargin': 0,
    "font.family": "serif",
    "font.serif": "cmr10",
    "text.usetex": False,  # Optional: if you want real LaTeX rendering
    'axes.labelsize': 14,
    "mathtext.fontset": "cm",    # Use Computer Modern for math symbols
    'axes.titlesize': 18,
    'axes.unicode_minus': False,
})


textwidth_pt = 426.79135
inches_per_pt = 1 / 72.27
TEXT_WIDTH = textwidth_pt * inches_per_pt  # in inches
fig_height = TEXT_WIDTH * 0.6  # for a 3:2 aspect ratio, adjust as needed

MARKERS = ['o', 's', '^', 'D', 'v', '<', 'p', '*','>'  ]
MODELS = [
"LogReg",
"MLP",
"LightGBM",
"LinReg",
"TabTransformer",
"TabNet",
"ResNet",
"FTTransformer",
]
MODEL_TO_MARKER = {model: marker for model, marker in zip(MODELS, MARKERS)}
COLORMAP = cc.glasbey_light
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
real_world_clf = sorted(list(set(datasets_clf + categorical_datasets_clf)))

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
 'syn polynomial \n(d:20, inf f.:5, noise:0.1)',
 'syn exponential_interaction \n(d:50, inf f.:2, noise:0.2)',
 'syn linear \n(d:50, inf f.:2, noise:0.6)',
 ]

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
    "Accuracy const. local model": 15,
    "Variance prob.": 12, 
    "Variance logit": 13,
    "Radius": 14,
    "Local Ratio All Ones": 11,
    "Accuracy $g_x$ - Accuracy const. local model": (0, 15),
}

METRICS_MAP_REG = {
    "MSE $g_x$": 0,
    "MAE $g_x$": 1,
    "R2 kNN": 2,
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
                    path = get_kw_fp(files, kernel_width="default")
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
        legend_ax.legend(handles, labels, frameon=True, fontsize=11)
        method_title = method.split("/")[-1]
        method_title = " ".join(method_title.split("_"))
        if method == "lime" and lime_features == "all":
            method_title = method_title + " (all features)"
        title = f"{method_title.capitalize()} on {dataset.capitalize()}"
        if method == "lime" and lime_features == "all":
            title += " (all features)"
        y_position = 1.04 if "syn" in dataset else 1.02
        fig.suptitle(title, y=y_position)
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


def create_model_legend(ax, models, markers, bbox_to_anchor=(1.02, 0.5), plot_multiple=False):
    """Create a legend for models only, with specified markers."""
    if plot_multiple:
        ax.axis("off")
    model_handles = [
        plt.Line2D([], [], marker=markers[m], color='gray', label=m, linestyle='None')
        for m in models
    ]
    handles = [plt.Line2D([], [], color='none', label='Models:')] + model_handles
    legend = ax.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=bbox_to_anchor,
        frameon=False,
        ncol=1,
        fontsize=12,
        handletextpad=1.5,
        columnspacing=2.0,
        borderpad=1.0
    )
    for text in legend.get_texts():
        if text.get_text() == "Models:":
            text.set_weight('bold')
            text.set_fontsize(14)
    return legend

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
        ncol=2,
        fontsize=12,
        handletextpad=1.5,
        columnspacing=2.0,
        borderpad=1.0
    )
    for text in legend.get_texts():
        if text.get_text() in ["Datasets:", "Models:"]:
            text.set_weight('bold')
            text.set_fontsize(14)
    return legend


def edit_ticks(ticks, val, label, exclude_lower = -np.inf, exclude_upper=1.1):
    ticks = [t for t in ticks if exclude_lower<=t<exclude_upper]  # filter out > 1.1
    if exclude_lower not in ticks:
        ticks.append(exclude_lower)
    ticks = sorted(set(ticks))
    ticks_labels = [label if np.isclose(t, val) else str(round(t, 2)) for t in ticks]
    return ticks, ticks_labels
       
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
                               kernel_width="default",
                               plot_downsample_fraction=False,
                               plot_individual_random_seed=True,
                               complexity_regression = "best",
                               average_over_n_neighbors=200,
                               width = TEXT_WIDTH, 
                               height = TEXT_WIDTH * 0.9,
                               save=False):
    """Main plotting function comparing model complexity vs performance difference."""
    synthetic_dataset_name = get_synthetic_dataset_friendly_name_regression if regression else get_synthetic_dataset_friendly_name
    assert (plot_downsample_fraction and random_seed ==42) or (not plot_downsample_fraction), "Downsampled data is only available for random seed 42."
    create_legend_to_ax = ax is None
    if ax is None:
        ax = plt.subplots(figsize=(width, height))[1]
    cmap = COLOR_TO_CLF_DATASET if not regression else COLOR_TO_REG_DATASET
    res_dict = get_results_files_dict(method, 
                                      models, 
                                      datasets,
                                      distance, 
                                      random_seed=random_seed, 
                                      downsampled=plot_downsample_fraction,
                                      kernel_width=kernel_width)
    is_diff = "-" in metric
    is_ratio = "/" in metric    
    all_results = defaultdict(list)
    all_results_std = defaultdict(list)
    cutoff_at = 0 if (regression and is_ratio) else 0.5
    cutoff_value_replaced = -0.05 if regression and is_ratio else 0.45
    cutoff_label = f"$<${cutoff_at}"
    if summarizing_statistics is None:
        summarizing_statistics = lambda x, axis: np.nanmean(x, axis=axis)
    for model in models:
        model_results = res_dict.get(model, None)
        if model_results is None or len(model_results) == 0:
            print(f"No results found for {model}.")
            continue
        if type(random_seed) == int:
            random_seeds = [random_seed]
        else: 
            all_random_seeds = set()
            for dataset in datasets:
                files = model_results.get(synthetic_dataset_name(dataset), None)
                if files is None or len(files) == 0: continue
                random_seed_to_fp = get_random_seed_to_filepaths(model_results.get(synthetic_dataset_name(dataset), None))
                random_seeds = [int(rs_fp[0]) for rs_fp in random_seed_to_fp]
                all_random_seeds.update(random_seeds)
            random_seeds = sorted(list(all_random_seeds))
        gather_res_per_dataset = defaultdict(list)
        if plot_downsample_fraction:
            random_seeds = [42]
            downsample_fractions = np.linspace(0.5, 1.0, 10)
        else:
            downsample_fractions = [None]
        for downsample_fraction in downsample_fractions:
            for rs in random_seeds: 
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
                                                kernel_width = kernel_width, 
                                                difference_to_constant_model=difference_to_constant_model,
                                                random_seed=rs,
                                                downsample_fraction=downsample_fraction,
                                                complexity_regression=complexity_regression,)
                if type(random_seed) == int or plot_individual_random_seed:
                    for dataset, knn_metric_res, filtered_res in points:
                        all_results[model].append((dataset, knn_metric_res, filtered_res))
                else:
                    for dataset, knn_metric_res, filtered_res in points:
                        gather_res_per_dataset[dataset].append((knn_metric_res, filtered_res))
        for dataset, knn_metric_res_list in gather_res_per_dataset.items():
            knn_metric_res = np.nanmean([res[0] for res in knn_metric_res_list])
            filtered_res = np.nanmean([res[1] for res in knn_metric_res_list])
            knn_metric_res_std = np.nanstd([res[0] for res in knn_metric_res_list])
            filtered_res_std = np.nanstd([res[1] for res in knn_metric_res_list])
            all_results[model].append((dataset, knn_metric_res, filtered_res))
            all_results_std[model].append((dataset, knn_metric_res_std, filtered_res_std))
    unique_datasets = sorted({d for m in all_results for d, _, _ in all_results[m]}, key=lambda x: extract_sort_keys(x, regression))
    colors = {d: cmap[d] for i, d in enumerate(unique_datasets)}
    models_plotted = list(all_results.keys())
    markers_to_models = {m: MODEL_TO_MARKER[m] for i, m in enumerate(models_plotted)}
    current_min_x_y = 1
    for i, model in enumerate(models_plotted):
        x_vals, y_vals, colors_list = [], [], []
        for dataset, knn_metric_res, filtered_res in all_results[model]:
            x_vals.append(knn_metric_res)
            y_vals.append(filtered_res if filtered_res > cutoff_at else cutoff_value_replaced)
            colors_list.append(colors[dataset])
        current_min_x_y = min(current_min_x_y, min(x_vals), min(y_vals))
        ax.scatter(x_vals, y_vals, c=colors_list, marker=markers_to_models[model], 
               s=80, alpha=1, edgecolors=colors_list, linewidths=0.5)
        if not(type(random_seed)==int or plot_individual_random_seed):
            x_vals_std, y_vals_std = [], []
            for dataset, knn_metric_res_std, filtered_res_std in all_results_std[model]:
                x_vals_std.append(knn_metric_res_std)
                y_vals_std.append(filtered_res_std)
            for i, (xv, yv, xv_std, yv_std) in enumerate(zip(x_vals, y_vals, x_vals_std, y_vals_std)):
                if yv > cutoff_at:
                    ellipse = plt.matplotlib.patches.Ellipse(
                    (xv, yv), width=2*xv_std, height=2*yv_std,
                    edgecolor='none', facecolor=colors_list[i], alpha=0.15, zorder=1
                    )
                    ax.add_patch(ellipse)
                else:# Plot horizontal error bar for x-value standard deviation
                    ax.errorbar(xv, yv, xerr=xv_std, fmt='none', ecolor=colors_list[i], alpha=0.3, capsize=3, zorder=1)
               

    # === Handle axes ===
    if is_diff or "R2" in metric:
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
    elif is_ratio:
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        # ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
    elif "Accuracy" in metric:
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.7)
    
    current_min_x_y = np.max([cutoff_value_replaced, current_min_x_y])
    current_min_x_y = np.min([cutoff_at, current_min_x_y])
    
    # === Labels ===
    metric_axis_label = "$R^2$" if regression else "accuracy"
    if complexity_regression == "best":
        # smpl_model = "one of: kNN-reg., linear reg. or decision tree\n" if regression else "one of: kNN-clf., logistic reg. or decision tree\n"
        smpl_model = "baseline models"
    elif complexity_regression == "kNN":
        smpl_model = "kNN clf." if not regression else "kNN reg."
    elif complexity_regression == "linear":
        smpl_model = "Lin. Reg" if regression else "Log. Reg"
    elif complexity_regression == "tree":
        smpl_model = "Decision Tree"
    ax.set_xlabel(f"{"Best " if complexity_regression == "best" else ""}{"$R^2$" if regression else "accuracy"} of {smpl_model} on model predictions")
    method_title = ' '.join(method.split('/')[-1].split('_')) 
    if method == "lime":
        method_title+= f" (sparse feat. space, kernel: {kernel_width if kernel_width == 'default' else kernel_width+' of default'})"
    if method == "lime_captum":
        method_title += f" (cont. feat. space, kernel: {kernel_width if kernel_width == 'default' else kernel_width+' of default'})"
    y_axis_label = f"Best avg. {metric_axis_label} of $g_x$" #get_y_axis_label(filter, metric, is_diff, is_ratio, summary=summarizing_statistics)
    ax.set_ylabel(y_axis_label)
    ax.set_xlim((0, 1)) if regression else  ax.set_xlim((0.5, 1))
    ax.set_ylim((current_min_x_y, 1))
    yticks = ax.get_yticks()
    yticks, ytick_labels = edit_ticks(yticks, cutoff_value_replaced, cutoff_label, exclude_lower=current_min_x_y, exclude_upper=1.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    if isinstance(filter, int):
        ax.set_title(f"{method_title.capitalize()}-\n Complexity of f vs. {metric_axis_label} avg. {'improvement' if is_diff else 'of $g_x$'} within {filter} neighbors")
    else:
        ax.set_title(f"{method_title.capitalize()}-\n Complexity of f vs. {filter} avg. {metric_axis_label} {'improvement' if is_diff else 'of $g_x$'}")
    ax.grid(True, alpha=0.3)
    ax.set_position([
        ax.get_position().x0,
        ax.get_position().y0,
        ax.get_position().width * 0.8,
        ax.get_position().height
    ])
    if create_legend_to_ax:
        create_dual_legend(ax, unique_datasets, colors, models_plotted,markers_to_models, bbox_to_anchor=(1.02, 0.5))

    if save:
        plt.savefig(
            f"graphics/complexity_vs_{filter}_metrics_{method.split('/')[-1]}_{dataset}.pdf",
            bbox_inches='tight',
            dpi=100,
            metadata={'CreationDate': None}
        )
    return ax, unique_datasets, colors, models_plotted, markers_to_models

def extract_random_seed(file_path):
    match = re.search(r'random_seed-(\d+)', file_path)
    return float(match.group(1)) if match else float('inf')

def order_by_seed(file_paths):
    """Order file paths by the downsampling fraction in the string."""
    return sorted(file_paths, key=extract_random_seed)

def plot_random_seeds_results_per_dataset(method, 
                                    models, 
                                    datasets, 
                                    regression=False):
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

    results_dict = get_results_files_dict(method, models, datasets, downsampled=False, random_seed=42)

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
        ax.set_ylabel(f"{'MSE' if regression else 'accuracy'} of $g_x$", fontsize=12)
        ax.set_title(f"{method.split("/")[-1].capitalize()}: Averaged over 5 random seeds, {'MSE' if regression else 'accuracy'} vs. Neighborhood Size\nDataset: {dataset}", 
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
                           random_seed=42,
                           kernel_width="default",
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
        rs_files = get_random_seed_to_filepaths(files)
        if len(rs_files) == 0: continue
        random_seeds = np.array([int(rs[0])for rs in rs_files])
        files_sorted_with_rs = [str(rs[1]) for rs in rs_files]
        try: 
            files_random_seed = files_sorted_with_rs[np.where(random_seeds==random_seed)[0][0]]
        except IndexError:
            print(f"{files}: Random seed {random_seed} not found in {dataset}.")
            continue
        file_path = get_kw_fp(files_random_seed, kernel_width)
        data, _ = load_results(file_path)
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
                               width = TEXT_WIDTH, 
                               kernel_width="default",
                               random_seed=42,
                               plot_individual_random_seed=True,
                               height = TEXT_WIDTH * 0.6,
                               save=False):
    """Main plotting function comparing model complexity vs performance difference."""
    synthetic_dataset_name = get_synthetic_dataset_friendly_name_regression if regression else get_synthetic_dataset_friendly_name
    
    create_legend_to_ax = ax is None
    if ax is None:
        ax = plt.subplots(figsize=(width, height))[1]
    cmap = COLOR_TO_REG_DATASET if regression else COLOR_TO_CLF_DATASET
    res_dict = get_results_files_dict(method, models, datasets, distance, kernel_width= kernel_width, random_seed=random_seed)
    is_diff = "-" in metric
    is_ratio = "/" in metric    
    cut_off = 0 if regression else 0.5
    replace_with = -0.05 if regression else 0.45
    all_results = defaultdict(list)
    all_results_std = defaultdict(list)

    if summarizing_statistics is None:
        summarizing_statistics = lambda x, axis: np.nanmean(x, axis=axis)

    for model in models:
        model_results = res_dict.get(model, None)
        if model_results is None or len(model_results) == 0:
            print(f"No results found for {model}.")
            continue
        if type(random_seed) == int:
            random_seeds = [random_seed]
        else: 
            all_random_seeds = set()
            for dataset in datasets:
                files = model_results.get(synthetic_dataset_name(dataset), None)
                if files is None or len(files) == 0: continue
                random_seed_to_fp = get_random_seed_to_filepaths(model_results.get(synthetic_dataset_name(dataset), None))
                random_seeds = [int(rs_fp[0]) for rs_fp in random_seed_to_fp]
                all_random_seeds.update(random_seeds)
            random_seeds = sorted(list(all_random_seeds))
        gather_res_per_dataset = defaultdict(list)
        for rs in random_seeds: 
            points = get_local_vs_constant_metric_data(res_model=model_results,
                           filter=filter,
                            metric=metric,
                            regression=regression, 
                            random_seed=rs,
                            kernel_width=kernel_width,
                            summarizing_statistics=summarizing_statistics,
                            average_over_n_neighbors=average_over_n_neighbors)
            if type(random_seed) == int or plot_individual_random_seed:
                for dataset, knn_metric_res, filtered_res in points:
                    all_results[model].append((dataset, knn_metric_res, filtered_res))
            else:
                for dataset, knn_metric_res, filtered_res in points:
                    gather_res_per_dataset[dataset].append((knn_metric_res, filtered_res))
        for dataset, knn_metric_res_list in gather_res_per_dataset.items():
            knn_metric_res = np.nanmean([res[0] for res in knn_metric_res_list])
            filtered_res = np.nanmean([res[1] for res in knn_metric_res_list])
            knn_metric_res_std = np.nanstd([res[0] for res in knn_metric_res_list])
            filtered_res_std = np.nanstd([res[1] for res in knn_metric_res_list])
            all_results[model].append((dataset, knn_metric_res, filtered_res))
            all_results_std[model].append((dataset, knn_metric_res_std, filtered_res_std))

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
        if not(type(random_seed)==int or plot_individual_random_seed):
            x_vals_std, y_vals_std = [], []
            for dataset, knn_metric_res_std, filtered_res_std in all_results_std[model]:
                x_vals_std.append(knn_metric_res_std)
                y_vals_std.append(filtered_res_std)
            for i, (xv, yv, xv_std, yv_std) in enumerate(zip(x_vals, y_vals, x_vals_std, y_vals_std)):
                if yv > cut_off and xv > cut_off:
                    ellipse = plt.matplotlib.patches.Ellipse(
                    (xv, yv), width=2*xv_std, height=2*yv_std,
                    edgecolor='none', facecolor=colors_list[i], alpha=0.15, zorder=1
                    )
                    ax.add_patch(ellipse)
               
       
    # === Handle axes ===
    if is_diff or "R2" in metric:
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax.set_ylim((replace_with, 1))
    elif is_ratio:
        ax.plot([current_min_x_y, 1], [current_min_x_y, 1], linestyle='--', color='gray', alpha=0.7)
        ax.set_ylim((current_min_x_y, 1))
    elif "Accuracy" in metric:
        current_min_x_y = np.max([replace_with, current_min_x_y])
        current_min_x_y = np.min([cut_off, current_min_x_y])
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.7)
        ax.axvline(0.5, color='black', linestyle='--', alpha=0.7)
        ax.plot([current_min_x_y, 1], [current_min_x_y, 1], linestyle='--', color='gray', alpha=0.7)
        ax.set_ylim((current_min_x_y, 1))
    ax.set_xlim((current_min_x_y, 1))
    yticks = ax.get_yticks()
    yticks, ytick_labels = edit_ticks(yticks, replace_with, f"$<${cut_off}", exclude_lower=current_min_x_y, exclude_upper=1.1)
    ax.set_yticks(sorted(set(yticks)))
    ax.set_yticklabels(ytick_labels)
    xticks = ax.get_xticks()
    xticks, xtick_labels = edit_ticks(xticks, replace_with, f"$<${cut_off}", exclude_lower=current_min_x_y, exclude_upper=1.1)
    ax.set_xticks(sorted(set(xticks)))
    ax.set_xticklabels(xtick_labels)
    ax.set_aspect('equal', adjustable='box')
    # # === Labels ===
    metric_axis_label = "$R^2$" if regression else "accuracy"
    method_title = ' '.join(method.split('/')[-1].split('_'))
    y_axis_label = f"Best avg. {metric_axis_label} of $g_x$" if isinstance(filter, str) else f"Average {metric_axis_label} $g_x$ for {filter} closest neighbors"
    x_axis_label = f"Best avg. {metric_axis_label} of const. local model" if isinstance(filter, str) else f"Average {metric_axis_label} const. local model for {filter} closest neighbors"
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)

    # === Title ===
    if isinstance(filter, int):
        ax.set_title(f"{method_title.capitalize()+": " if create_legend_to_ax else ""}Average {metric_axis_label} of const. local model vs.\n {metric_axis_label} avg. of $g_x$ within {filter} neighbors")
    else:
        ax.set_title(f"{method_title.capitalize()+": " if create_legend_to_ax else ""}Max. avg. {metric_axis_label} of const. local model vs.\n  max. avg. {metric_axis_label} of $g_x$")

    ax.grid(True, alpha=0.3)
    if create_legend_to_ax:
        create_dual_legend(ax, unique_datasets, colors, models_plotted, markers_to_models)

    if save:
        plt.savefig(
            f"graphics/complexity_vs_{filter}_metrics_{method.split('/')[-1]}_{dataset}.pdf",
            bbox_inches='tight',
            dpi=100,
            metadata={'CreationDate': None}
        )
    return ax, unique_datasets, colors, models_plotted, markers_to_models

def plot_knn_vs_model_performance_scatter(models, 
                        datasets, 
                        distance="euclidean", 
                        regression=False, 
                        synthetic=False, 
                        random_seed=42,
                        complexity_regression="best", 
                        ax=None, 
                        width=TEXT_WIDTH,
                        height=TEXT_WIDTH * 0.6,
                        save=False):
    """
    Scatter plot: x-axis = performance_smple_model_model_preds, y-axis = diff (model - simple model on true labels)
    Each point: (model, dataset). Color by dataset, marker by model.
    """
    
    create_legend_to_ax = ax is None
    
    if regression:
        from src.utils.process_results import get_synthetic_dataset_friendly_name_regression as get_friendly_name
    else:
        from src.utils.process_results import get_synthetic_dataset_friendly_name as get_friendly_name
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure
    from src.utils.plotting_utils import COLOR_TO_CLF_DATASET, COLOR_TO_REG_DATASET, extract_sort_keys
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    cmap = COLOR_TO_REG_DATASET if regression else COLOR_TO_CLF_DATASET
    results = []
    for model in models:
        for dataset in datasets:
            try:
                _, x, y = get_performance_metrics_smpl_complex_models(model, 
                                                                      dataset, 
                                                                      distance=distance, 
                                                                      regression=regression,
                                                                      synthetic=synthetic, 
                                                                      random_seed=random_seed, 
                                                                      complexity_regression=complexity_regression)
                results.append((model, get_friendly_name(dataset), x, y))
            except Exception as e:
                print(f"Skipping {model} on {dataset}: {e}")

    unique_datasets = sorted({d for _, d, _, _ in results}, key=lambda x: extract_sort_keys(x, regression))
    colors = {d: cmap[d] for d in unique_datasets}
    models_plotted = sorted({m for m, _, _, _ in results})
    markers_to_models = {m: MODEL_TO_MARKER[m] for i, m in enumerate(models_plotted)}

    for model, dataset, x, y in results:
        ax.scatter(x, y, color=colors[dataset], marker=markers_to_models[model], s=80, alpha=0.8)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
    # smpl_model = "Lin. Reg" if regression and complexity_regression else ("kNN-reg." if regression else ("Log. Reg" if complexity_regression else "kNN-clf."))
    smpl_model = "Lin. Reg" if regression and complexity_regression else ("kNN-reg." if regression else ("Log. Reg" if complexity_regression else "kNN-clf."))
    if complexity_regression == "best":
        # smpl_model = "one of: kNN-reg., linear reg. or decision tree\n" if regression else "one of: kNN-clf., logistic reg. or decision tree\n"
        smpl_model = "baseline models"
    elif complexity_regression == "kNN":
        smpl_model = "kNN clf." if not regression else "kNN reg."
    elif complexity_regression == "linear":
        smpl_model = "Lin. Reg" if regression else "Log. Reg"
    elif complexity_regression == "tree":
        smpl_model = "Decision Tree"
    else:
        smpl_model = "smpl model"
    ax.set_xlabel(f"{"Best " if complexity_regression == "best" else ""}{"$R^2$" if regression else "accuracy"} of {smpl_model} on true labels")
    ax.set_ylabel(f"{"$R^2$" if regression else "accuracy"} of f on true labels")
    ax.set_title(f"Complex models vs.\n baseline models on true labels")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_ylim((0, 1)) if regression else ax.set_ylim((0.5, 1))
    ax.set_xlim((0, 1)) if regression else ax.set_xlim((0.5, 1))
    
    if create_legend_to_ax:
        create_dual_legend(ax, unique_datasets, colors, models_plotted, markers_to_models)


    plt.tight_layout()
    if save:
        plt.savefig("graphics/knn_vs_diff_scatter.pdf", bbox_inches='tight', dpi=100, metadata={'CreationDate': None})
    return ax, unique_datasets, colors, models_plotted, markers_to_models

def plot_knn_vs_diff_scatter(models, 
                        datasets, 
                        distance="euclidean", 
                        regression=False, 
                        synthetic=False, 
                        random_seed=42,
                        complexity_regression="best", 
                        ax=None, 
                        width=TEXT_WIDTH,
                        height=TEXT_WIDTH * 0.6,
                        save=False):
    """
    Scatter plot: x-axis = performance_smple_model_model_preds, y-axis = diff (model - simple model on true labels)
    Each point: (model, dataset). Color by dataset, marker by model.
    """
    create_legend_to_ax = ax is None
    if regression:
        from src.utils.process_results import get_synthetic_dataset_friendly_name_regression as get_friendly_name
    else:
        from src.utils.process_results import get_synthetic_dataset_friendly_name as get_friendly_name
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure

    # Prepare colors and markers
    from src.utils.plotting_utils import COLOR_TO_CLF_DATASET, COLOR_TO_REG_DATASET, extract_sort_keys
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    cmap = COLOR_TO_REG_DATASET if regression else COLOR_TO_CLF_DATASET
    current_min_x = np.inf
    cutoff = 0 if regression else 0.5
    # Gather results
    results = []
    for model in models:
        for dataset in datasets:
            try:
                x, y = get_knn_vs_diff_model_performance(
                    model, dataset, distance=distance, regression=regression,
                    synthetic=synthetic, random_seed=random_seed, complexity_regression=complexity_regression
                )
                current_min_x = min(current_min_x, x)
                results.append((model, get_friendly_name(dataset), x, y))
            except Exception as e:
                print(f"Skipping {model} on {dataset}: {e}")

    unique_datasets = sorted({d for _, d, _, _ in results}, key=lambda x: extract_sort_keys(x, regression))
    colors = {d: cmap[d] for d in unique_datasets}
    models_plotted = sorted({m for m, _, _, _ in results})
    markers_to_models = {m: markers[i % len(markers)] for i, m in enumerate(models_plotted)}

    for model, dataset, x, y in results:
        if x<cutoff: continue
        ax.scatter(x, y, color=colors[dataset], marker=markers_to_models[model], s=80, alpha=0.8)
    smpl_model = "Lin. Reg" if regression and complexity_regression else ("kNN-reg." if regression else ("Log. Reg" if complexity_regression else "kNN-clf."))
    if complexity_regression == "best":
        # smpl_model = "one of: kNN-reg., linear reg. or decision tree\n" if regression else "one of: kNN-clf., logistic reg. or decision tree\n"
        smpl_model = "baseline models"
    elif complexity_regression == "kNN":
        smpl_model = "kNN clf." if not regression else "kNN reg."
    elif complexity_regression == "linear":
        smpl_model = "Lin. Reg" if regression else "Log. Reg"
    elif complexity_regression == "tree":
        smpl_model = "Decision Tree"
    else:
        smpl_model = "smpl model"  
    ax.axhline(0, color='black', linestyle='--', alpha=0.7)  
    ax.set_xlabel(f"{"Best " if complexity_regression == "best" else ""}{"$R^2$" if regression else "accuracy"} of {smpl_model} on model predictions")
    ax.set_ylabel(f"Difference in {"$R^2$" if regression else "accuracy"} on true labels")
    ax.set_title(f"Complexity of $f$ vs.\n Difference of complex and baseline models on true labels")
    ax.grid(True, alpha=0.3)
    ax.set_xlim((cutoff, 1))
    ax.set_aspect('equal', adjustable='box')

    if create_legend_to_ax:
        create_dual_legend(ax, unique_datasets, colors, models_plotted, markers_to_models)

    plt.tight_layout()
    if save:
        plt.savefig("graphics/knn_vs_diff_scatter.pdf", bbox_inches='tight', dpi=100, metadata={'CreationDate': None})
    return ax, unique_datasets, colors, models_plotted, markers_to_models