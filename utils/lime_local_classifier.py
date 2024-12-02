import numpy as np
import lime
from sklearn.metrics import pairwise_distances

def get_feat_coeff_intercept(exp):
    top1_model = next(iter(exp.local_pred))
    feat_ids = []
    coeffs = []
    for feat_id, coeff in exp.local_exp[top1_model]:
        feat_ids.append(feat_id)
        coeffs.append(coeff)
    return feat_ids, np.array(coeffs), exp.intercept[top1_model]

def get_binary_vector(x, x_explained, explainer):
    bins = explainer.discretizer.discretize(x)
    bins_instance = explainer.discretizer.discretize(x_explained)
    binary = (bins == bins_instance).astype(int)
    return binary


def lime_pred(binary_x, exp):
    if binary_x.ndim == 1:
        binary_x = binary_x.reshape(1, -1)
    feat_ids, coeffs, intercept = get_feat_coeff_intercept(exp)
    local_pred = intercept + np.dot(binary_x[:, feat_ids], coeffs)
    return local_pred


def binary_pred(pred, threshold):
    return (pred >= threshold).astype(int)


def get_sample_close_to_x(x, dataset, dist_threshold, distance_measure="cosine"):
    distances = pairwise_distances(x.reshape(1, -1), dataset, distance_measure)[0]
    if distance_measure == "cosine":
        distances = 1 - distances
    max_min_dist = np.max(distances[distances < dist_threshold])
    return dataset[distances < dist_threshold], max_min_dist


def compute_lime_accuracy(x, dataset, explainer, predict_fn,  dist_measure, dist_threshold, tree = None, pred_threshold=None):
    """
    Computes the accuracy of a LIME explanation by comparing the local model's predictions
    to the original model's predictions for samples within a ball around the instance.

    Args:
        x (numpy.ndarray): The instance to explain, a 1D numpy array.
        dataset (numpy.ndarray): The dataset to sample from, a 2D numpy array.
        explainer (lime.lime_tabular.LimeTabularExplainer): The LIME explainer object.
        predict_fn (callable): The prediction function suitable for LIME.
        dist_measure (str): sklearn pariwise distance measure to use (e.g., "cosine").
        dist_threshold (float): The distance threshold to determine closeness.
        pred_threshold (float, optional): The threshold for binary classification. Defaults to 0.5.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): The accuracy of the local model's predictions.
            - ratio_within_ball (float): The ratio of samples within the distance threshold.
            - radius (float): The maximum distance of samples within the threshold.
            - num_samples (int): The number of samples within the distance threshold.
            - ratio_all_ones (float): The fraction of samples that are discretized into all 1s. 
    """

    exp = explainer.explain_instance(x, predict_fn, top_labels=1)
    if tree is not None:
        idx, dist = tree.query_radius(x.reshape(1,-1), dist_threshold, count_only=False, return_distance = True)
        radius = np.max(dist[0])
        samples_in_ball = dataset[idx[0]]
    else: 
        samples_in_ball, radius = get_sample_close_to_x(x, dataset, dist_threshold, dist_measure)
    fraction_points_in_ball = len(samples_in_ball)/len(dataset)
    
    binary_sample = get_binary_vector(x, samples_in_ball, explainer)
    ratio_all_ones = np.all(binary_sample, axis=1).sum()/len(binary_sample)
    model_pred = predict_fn(samples_in_ball)
    local_pred = lime_pred(binary_sample, exp)
    if pred_threshold is None:
        pred_threshold = 0.5
    local_classification = binary_pred(local_pred, pred_threshold)
    model_classification = np.argmax(model_pred, axis=1)
    accuracy = np.sum(local_classification == model_classification)/len(samples_in_ball)
    return accuracy, fraction_points_in_ball, radius, len(samples_in_ball), ratio_all_ones

