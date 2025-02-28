from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def regression_metrics_per_row(y_true, y_pred):
    return mean_squared_error(y_true.T, y_pred.T, multioutput='raw_values'), mean_absolute_error(y_true.T, y_pred.T, multioutput='raw_values'), r2_score(y_true.T, y_pred.T, multioutput='raw_values')


def binary_classification_metrics(y_true, y_pred_prob, prediction_threshold):
    """
    Compute binary classification metrics for a single row.
    
    Parameters:
    y_true : numpy.ndarray
        Shape (k, C) containing true labels for a single row.
    y_pred_prob : numpy.ndarray
        Shape (k, C) containing predicted probabilities for a single row.
    prediction_threshold : float
        Threshold for converting probabilities to binary predictions.
    
    Returns:
    tuple
        A tuple containing AUROC, accuracy, precision, recall, and F1 score.
    """
    y_pred = (y_pred_prob > prediction_threshold).astype(int)
    if len(y_true) == 1:
        aucroc = 0
    else:     
        try:
            aucroc = roc_auc_score(y_true, y_pred_prob)
        except ValueError:
            aucroc = None
    return (
        aucroc,
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0)
    )

def binary_classification_metrics_per_row(y_true, y_pred_prob, prediction_threshold):
    """
    Vectorized computation of classification metrics.
    """
    metrics = np.array([
        binary_classification_metrics(y_true[i], y_pred_prob[i], prediction_threshold)
        for i in range(y_true.shape[0])
    ])
    
    return tuple(metrics.T)

def fractions_of_ones(y_true):
    return np.sum(y_true, axis=-1) / y_true.shape[-1]


def impurity_metrics_per_row(y_label):
    """
    Compute impurity metrics for a single row: Gini impurity and fraction of ones.
    
    Parameters:
    y_true : numpy.ndarray
        Shape (k, C) containing true labels for a single row.
    Returns:
    tuple
        A tuple containing Gini impurity and fraction of ones, aka the model predicting the same label as for instance x
    """
    return (
        compute_gini_impurity_vectorized(y_label),
        fractions_of_ones(y_label),
    )
def gini_impurity(labels):
    """
    Compute Gini impurity for a single row of predictions.
    
    Parameters:
    labels : numpy.ndarray
        Shape (k_neighbors,) containing class predictions for a single test sample.
    
    Returns:
    float
        Gini impurity for the row.
    """
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return 1 - np.sum(probs ** 2)

def compute_gini_impurity_vectorized(labels_model_preds):
    """
    Compute Gini impurity for each row in the input array using vectorization.
    
    Parameters:
    labels_model_preds : numpy.ndarray
        Shape (n_test_samples, k_neighbors) containing class predictions.
    
    Returns:
    numpy.ndarray
        Shape (n_test_samples,) containing Gini impurity for each test sample.
    """
    return np.apply_along_axis(gini_impurity, axis=1, arr=labels_model_preds)


