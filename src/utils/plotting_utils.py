import numpy as np
import matplotlib.pyplot as plt
# from src.explanation_methods.lime_analysis.lime_local_classifier import get_feat_coeff_intercept
from matplotlib.lines import Line2D
from matplotlib import cm
from typing import Optional

light_red = "#ffa5b3"
light_blue = "#9cdbfb"
light_grey = "#61cff2"

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
