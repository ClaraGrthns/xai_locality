import numpy as np
import lime
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed


def get_feat_coeff_intercept(exp):
    """
    Extracts the feature IDs, coefficients, and intercept from a LIME explanation.

    Args:
        exp (lime.explanation.Explanation): The LIME explanation object.

    Returns:
        tuple: A tuple containing:
            - feat_ids (list): List of feature IDs.
            - coeffs (numpy.ndarray): Array of coefficients corresponding to the feature IDs.
            - intercept (float): The intercept of the local linear model.
    """
    top1_model = exp.top_labels[0]
    feat_ids = []
    coeffs = []
    for feat_id, coeff in exp.local_exp[top1_model]:
        feat_ids.append(feat_id)
        coeffs.append(coeff)
    return np.array(feat_ids), np.array(coeffs), exp.intercept[top1_model]

def get_binary_vector(samples_around_xs:list , xs:np.array, explainer):
    """
    Converts the features of the instance and the explained instance into binary vectors.

    Args:
        samples_around_xs (list): samples around the instance, a list of 2D numpy arrays.
        xs (numpy.ndarray): The explained instance, a 1D numpy array.
        explainer (lime.lime_tabular.LimeTabularExplainer): The LIME explainer object.

    Returns:
        numpy.ndarray: A binary vector indicating which features match between the instance and the explained instance.
    """
    bins_sample = [explainer.discretizer.discretize(sample_in_ball) for sample_in_ball in samples_around_xs]
    del samples_around_xs
    bins_instance = explainer.discretizer.discretize(xs)
    binary = [(bins_sample[i] == bins_instance[i]).astype(int) for i in range(len(bins_sample))]
    del bins_sample
    del bins_instance
    return binary

def get_binary_vectorized(samples_around_xs:list , xs:np.array, explainer):
    """
    Converts the features of the instance and the explained instance into binary vectors.

    Args:
        samples_around_xs (list): samples around the instance, a list of 2D numpy arrays.
        xs (numpy.ndarray): The explained instance, a 1D numpy array.
        explainer (lime.lime_tabular.LimeTabularExplainer): The LIME explainer object.

    Returns:
        numpy.ndarray: A binary vector indicating which features match between the instance and the explained instance.
    """
    bins_sample = explainer.discretizer.discretize(samples_around_xs)
    bins_instance = explainer.discretizer.discretize(xs)
    binary = (bins_sample == bins_instance[:, None, :])
    return binary

def lime_pred_vectorized(binary_xs, exps):
    """
    Computes the local prediction using the binary vector and the LIME explanation.

    Args:
        binary_x (numpy.ndarray): The binary vector representing the instance.
        exp (lime.explanation.Explanation): The LIME explanation object.

    Returns:
        numpy.ndarray: The local prediction for the instance.
    """
    feat_ids_array = []
    coeffs_array = []
    intercept_array = []

    for exp in exps:
        feat_ids, coeffs, intercept = get_feat_coeff_intercept(exp)
        feat_ids_array.append(feat_ids)
        coeffs_array.append(coeffs)
        intercept_array.append(intercept)

    feat_ids_array = np.array(feat_ids_array)[:, None, :]
    coeffs_array = np.array(coeffs_array)[:, None, :]
    intercept_array = np.array(intercept_array)
        
    # Pre-select only needed features before dot product
    binary_x_selected = binary_xs[np.arange(binary_xs.shape[0])[:, None, None], np.arange(binary_xs.shape[1])[None, :, None], feat_ids_array]
    local_pred = intercept_array[:, None] + np.matmul(binary_x_selected, coeffs_array.transpose(0, 2, 1)).squeeze(-1)
    return local_pred

def lime_pred(binary_x, exp):
    """
    Computes the local prediction using the binary vector and the LIME explanation.

    Args:
        binary_x (numpy.ndarray): The binary vector representing the instance.
        exp (lime.explanation.Explanation): The LIME explanation object.

    Returns:
        numpy.ndarray: The local prediction for the instance.
    """
    if binary_x.ndim == 1:
        binary_x = binary_x.reshape(1, -1)
    feat_ids, coeffs, intercept = get_feat_coeff_intercept(exp)
    # Pre-select only needed features before dot product
    binary_x_selected = binary_x[:, feat_ids]
    local_pred = intercept + binary_x_selected @ coeffs
    return local_pred

def compute_fractions(thresholds, tst_feat, df_feat, tree):
    def compute_fraction_for_threshold(threshold):
        counts = tree.query_radius(tst_feat, r=threshold, count_only=True)
        return counts / df_feat.shape[0]

    fraction_points_in_ball = Parallel(n_jobs=-1)(
        delayed(compute_fraction_for_threshold)(threshold) for threshold in thresholds
    )
    fraction_points_in_ball = np.array(fraction_points_in_ball)
    return fraction_points_in_ball

def compute_explanations(explainer, tst_feat, predict_fn, num_lime_features):
    enumerated_data = list(enumerate(tst_feat))
    
    # Run parallel computation with indices
    indexed_explanations = Parallel(n_jobs=-1)(
        delayed(lambda x: (x[0], explainer.explain_instance(x[1], predict_fn, top_labels=1, num_features = num_lime_features)))(item)
        for item in enumerated_data
    )
    sorted_explanations = [exp for _, exp in sorted(indexed_explanations, key=lambda x: x[0])]
    return sorted_explanations

def compute_lime_accuracy_per_radius(tst_set, dataset, explanations, explainer, predict_fn, dist_threshold, tree, pred_threshold=None):
    """
    Computes the accuracy of a LIME explanation by comparing the local model's predictions
    to the original model's predictions for samples within a ball around the instance.

    Args:
        x (numpy.ndarray): The instances to explain, a 2D numpy array.
        dataset (numpy.ndarray): The dataset to sample from, a 2D numpy array.
        explainer (lime.lime_tabular.LimeTabularExplainer): The LIME explainer object.
        predict_fn (callable): The prediction function suitable for LIME.
        dist_measure (str): sklearn pariwise distance measure to use (e.g., "cosine").
        dist_threshold (float): The distance threshold to determine closeness.
        pred_threshold (float, optional): The threshold for binary classification. Defaults to 0.5.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): The accuracy of the local model's predictions.
            - fraction_within_ball (float): The ratio of samples within the distance threshold.
            - num_samples (int): The number of samples within the distance threshold.
            - ratio_all_ones (float): The fraction of samples that are discretized into all 1s. 
    """
    if tst_set.ndim == 1:
        tst_set = tst_set.reshape(1, -1)

    len_test = len(tst_set)
    idx = tree.query_radius(tst_set, dist_threshold, count_only=False)
    samples_in_ball = [dataset[i] for i in idx]
    n_samples_in_ball = np.array([len(s) for s in samples_in_ball])
    
    fraction_points_in_ball = n_samples_in_ball / len(dataset) # output this seperately and save previous number of sampels in ball
    binary_sample = get_binary_vector(samples_in_ball, tst_set, explainer)
    ratio_all_ones = np.array([np.all(b, axis=1).sum() / len(b) for b in binary_sample])
    
    model_preds = [predict_fn(samples) for samples in samples_in_ball]
    local_preds = [lime_pred(b, explanations[i]) for i, b in enumerate(binary_sample)]
    if pred_threshold is None:
        pred_threshold = 1/model_preds[0].ndim
    top_labels = [exp.top_labels[0] for exp in explanations]
    local_detect_of_top_label = [(local_preds[i] > pred_threshold).astype(np.int32) for i in range(len(local_preds))]
    model_detect_of_top_label = [(np.argmax(pred, axis=1) == top_label).astype(np.int32)  for pred, top_label in zip(model_preds, top_labels)]
    accuracies_per_dp = np.array([np.mean(local_detect_of_top_label[i] == model_detect_of_top_label[i]) for i in range(len_test)])
    return dist_threshold, accuracies_per_dp, fraction_points_in_ball, n_samples_in_ball, ratio_all_ones

def compute_lime_accuracy_per_fraction(tst_set, dataset, explanations, explainer, predict_fn, n_closest, tree, pred_threshold=None):
    """
    Computes the accuracy of a LIME explanation by comparing the local model's predictions
    to the original model's predictions for samples within a ball around the instance.

    Args:
        x (numpy.ndarray): The instances to explain, a 2D numpy array.
        dataset (numpy.ndarray): The dataset to sample from, a 2D numpy array.
        explainer (lime.lime_tabular.LimeTabularExplainer): The LIME explainer object.
        predict_fn (callable): The prediction function suitable for LIME.
        dist_measure (str): sklearn pariwise distance measure to use (e.g., "cosine").
        dist_threshold (float): The distance threshold to determine closeness.
        pred_threshold (float, optional): The threshold for binary classification. Defaults to 0.5.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): The accuracy of the local model's predictions.
            - num_samples (int): The number of samples within the distance threshold.
            - ratio_all_ones (float): The fraction of samples that are discretized into all 1s. 
    """
    if tst_set.ndim == 1:
        tst_set = tst_set.reshape(1, -1)

    idx = tree.query(tst_set, k=n_closest, return_distance=False)

    samples_in_ball = dataset[idx]
    binary_sample = get_binary_vectorized(samples_in_ball, tst_set, explainer)
    ratio_all_ones = np.sum(np.all(binary_sample, axis=-1), axis=-1) / len(binary_sample)
    samples_reshaped = samples_in_ball.reshape(-1, samples_in_ball.shape[-1])
    model_preds = predict_fn(samples_reshaped)
    model_preds = model_preds.reshape(samples_in_ball.shape[0], samples_in_ball.shape[1], -1)
    # local_preds = np.array([lime_pred(b, explanations[i]) for i, b in enumerate(binary_sample)])
    local_preds = lime_pred_vectorized(binary_sample, explanations)
    if pred_threshold is None:
        pred_threshold = 1/model_preds[0].ndim
    top_labels = np.array([exp.top_labels[0] for exp in explanations])
    local_detect_of_top_label = local_preds>=  pred_threshold
    model_detect_of_top_label =  np.argmax(model_preds, axis=-1) == top_labels[:, None]
    accuracies_per_dp = np.mean(local_detect_of_top_label== model_detect_of_top_label, axis = -1)
    return n_closest, accuracies_per_dp, ratio_all_ones