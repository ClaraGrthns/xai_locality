from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from utils.lime_local_classifier import get_feat_coeff_intercept
def plot_3d_scatter(x, y, z, x_label="Fraction of points in ball", y_label="Thresholds", z_label="Accuracy", title="LIME Local Model on Test set", s=2, alpha=0.5, cmap='viridis', angles=(30, 40), save_figure=False):

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    y = np.repeat(y, x.shape[1])  # Flattened thresholds
    x = x.flatten()
    z = z.flatten()

    sc = ax.scatter(-x, y, z, c=z, s=s, alpha=alpha, cmap=cmap)  # Negated x to flip direction

    # Set labels
    ax.set_xlabel(x_label)
    ax.set_zlabel(z_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Set viewing angle (elevation, azimuth in degrees)
    # Default is elev=30, azim=-60
    ax.view_init(elev=angles[0], azim=angles[1])  # Changed azimuth to 120 to rotate view

    # Flip x-axis tick labels to show positive values
    ax.set_xticklabels([f'{abs(x):.2f}' for x in ax.get_xticks()])

    # Add colorbar
    cb = fig.colorbar(sc, ax=ax, label=z_label)

    plt.show()


def plot_3d_scatter_interactive(x, y, z, x_label="Fraction of points in ball", y_label="Thresholds", z_label="Accuracy", title="LIME Local Model on Test set", s=2, alpha=0.5, cmap='viridis'):
    z = np.repeat(z, x.shape[0])
    x = x.flatten()
    y = y.flatten()

    fig = go.Figure(data=go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=s,
            color=z,  # Use threshold values for color
            colorscale=cmap,  # Choose a color scale
            opacity=alpha
        )
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
        ),
        title=title,
    )
    fig.show()


def plot_lime_stats(explanations, plot_intercepts=True, plot_means_coefficients=True, model_predictions=None, title_add_on= ""):
    means_coefficients = [np.mean(get_feat_coeff_intercept(exp)[1]) for i, exp in enumerate(explanations)]
    intercepts = [get_feat_coeff_intercept(exp)[2] for i, exp in enumerate(explanations)]
    if plot_intercepts:
        # Plot vertical lines for intercepts
        colors = ['red' if y == 1 else 'blue' for y in model_predictions] if model_predictions is not None else ['blue'] * len(intercepts)
        plt.scatter(range(len(intercepts)), intercepts, s=1, c=colors)
        plt.title("LIME - Intercepts per test datapoint"+ " " + title_add_on) 
        plt.xlabel("Test datapoint")
        plt.ylabel("Intercept")
        
        # Add legend
        if model_predictions is not None:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='red', label='Prediction: 1', markersize=8),
                             Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='blue', label='Prediction: 0', markersize=8)]
            plt.legend(handles=legend_elements)
            
        plt.show()

    if plot_means_coefficients:
        #plot histogram of the means_coefficients
        plt.hist(means_coefficients, bins=100)
        plt.title("LIME - Means of the coefficients per test datapoint")
        plt.xlabel("Mean of the coefficients")
        plt.ylabel("Frequency")
        plt.show()
