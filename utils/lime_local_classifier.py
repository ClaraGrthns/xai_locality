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


def compute_lime_accuracy(x, dataset, explainer, predict_fn,  dist_measure, dist_threshold, pred_threshold=None):
    exp = explainer.explain_instance(x, predict_fn, top_labels=1)
    sample_closest_to_instance, radius = get_sample_close_to_x(x, dataset, dist_threshold, dist_measure)
    print(f"Number of samples close to the instance: {len(sample_closest_to_instance)}")
    binary_sample = get_binary_vector(x, sample_closest_to_instance, explainer)
    model_pred = predict_fn(sample_closest_to_instance)
    local_pred = lime_pred(binary_sample, exp)
    if pred_threshold is None:
        pred_threshold = 0.5
        print(f"Using the threshold {pred_threshold} for the prediction")
    local_classification = binary_pred(local_pred, pred_threshold)
    model_classification = np.argmax(model_pred, axis=1)
    ratio_within_ball = len(sample_closest_to_instance)/len(dataset)
    return np.sum(local_classification == model_classification)/len(sample_closest_to_instance), ratio_within_ball, radius
