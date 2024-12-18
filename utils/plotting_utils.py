from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from lime_analysis.lime_local_classifier import get_feat_coeff_intercept
from matplotlib.lines import Line2D

light_red = "#ffa5b3"
light_blue = "#9cdbfb"
light_grey = "#61cff2"

def plot_accuracy_vs_threshold(accuracy, 
                                thresholds, 
                                model_predictions, 
                                save_path = None):
    mean_accuracy = np.mean(accuracy, axis=1)   
    mean_accuracy_class_1 = np.mean(accuracy[:, model_predictions == 1], axis=1)
    mean_accuracy_class_0 = np.mean(accuracy[:, model_predictions == 0], axis=1)
    # Create figure and axis objects
    fig, ax = plt.subplots()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for dp in range(accuracy.shape[1]):
        color = light_red if model_predictions[dp] == 1 else light_blue
        ax.plot(thresholds, accuracy[:, dp], color=color, alpha=0.1)
    ax.set_xlabel("Thresholds")
    # add legend for color blue = 0, red = 1
    ax.plot(mean_accuracy, color='k', linestyle='dashed', linewidth=1, label = "mean accuracy")
    ax.plot(mean_accuracy_class_1, color='red', linestyle='solid', linewidth=2, label = "mean accuracy class 1")
    ax.plot(mean_accuracy_class_0, color='blue', linestyle='solid', linewidth=2, label = "mean accuracy class 0")

    legend_elements = [Line2D([0], [0], linestyle='solid', color='k', linewidth=1,
                markerfacecolor='black', label='Mean accuracy'),
                Line2D([0], [0], linestyle='solid', color='red',linewidth=1,
                markerfacecolor='red', label='Mean accuracy, pred: class 1'),
                Line2D([0], [0], linestyle='solid', color='blue', linewidth=1,
                markerfacecolor='blue', label='Mean accuracy, pred: class 0')]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.axhline(0.5, color='k', linestyle='dashed', linewidth=1)
    ax.set_ylabel("Accuracy")
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_title("Accuracy per threshold - kernel width 5.51")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_accuracy_vs_fraction(accuracy, 
                            fraction_points_in_ball, 
                            model_predictions, 
                            title_add_on="", 
                            save_path = None):
    mean_fraction = np.mean(fraction_points_in_ball, axis=1)
    mean_accuracy = np.mean(accuracy, axis=1)

    mean_accuracy_class_1 = np.mean(accuracy[:, model_predictions == 1], axis=1)
    mean_accuracy_class_0 = np.mean(accuracy[:, model_predictions == 0], axis=1)

    colors = [light_red if y == 1 else light_blue for y in model_predictions]
    color_array = np.repeat(colors, accuracy.T.shape[1], axis=0)

    # Create figure and axis objects
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.scatter(fraction_points_in_ball.T.flatten(), accuracy.T.flatten(), s=1.5, c=color_array)
    ax.scatter(mean_fraction, mean_accuracy, s=10, c='k', marker='x', label='Mean')
    ax.scatter(mean_fraction, mean_accuracy_class_1, s=10, c='red', marker='x', label='Mean accuracy, pred: class 1')
    ax.scatter(mean_fraction, mean_accuracy_class_0, s=10, c='blue', marker='x', label='Mean accuracy, pred: class 0')
    ax.axhline(0.5, color='k', linestyle='dashed', linewidth=1)
    ax.set_xlabel("Fraction of points in ball")
    ax.set_ylabel("Accuracy") 
    ax.set_title(f"Accuracy vs. Fraction of points in ball {title_add_on}")

    legend_elements = [
    Line2D([0], [0], marker='x', color='k', linestyle='None', label='Mean', markersize=6),
    Line2D([0], [0], marker='x', color='red', linestyle='None', label='Mean accuracy, pred: 1', markersize=6),
    Line2D([0], [0], marker='x', color='blue', linestyle='None', label='Mean accuracy, pred: 0', markersize=6),
    Line2D([0], [0], marker='o', color=light_red, linestyle='None', label='Prediction: 1', markersize=6),
    Line2D([0], [0], marker='o', color=light_blue, linestyle='None', label='Prediction: 0', markersize=6)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    
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
