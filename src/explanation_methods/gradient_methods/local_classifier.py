import numpy as np
from sklearn.neighbors import BallTree
import torch
import torch.nn.functional as F
from src.utils.metrics import binary_classification_metrics_per_row, regression_metrics_per_row, impurity_metrics_per_row

def linear_classifier(samples_in_ball, saliency_map):
    if samples_in_ball.ndim == 5:
        return torch.einsum('bcij, bkcij -> bk', saliency_map.float(), samples_in_ball)
    else:
        return torch.einsum('bc, bkc -> bk', saliency_map.float(), samples_in_ball)


def compute_gradmethod_accuracy_per_fraction(tst_feat, 
                                             top_labels, 
                                             analysis_data, 
                                             saliency_map, 
                                             predict_fn, 
                                             n_closest, 
                                             tree, 
                                             pred_threshold=None):
    if tst_feat.ndim == 1:
        tst_set = tst_set.reshape(1, -1)

    dist, idx= tree.query(tst_feat, k=n_closest, return_distance=True)
    R = np.max(dist, axis=-1)

    samples_in_ball = [[analysis_data[idx][0] for idx in row] for row in idx]
    samples_in_ball = torch.stack([torch.stack(row, dim=0) for row in samples_in_ball], dim=0)    
    samples_reshaped = samples_in_ball.reshape(-1, samples_in_ball.shape[2], samples_in_ball.shape[3], samples_in_ball.shape[4])

    model_preds = predict_fn(samples_reshaped)
    model_preds = model_preds.reshape(samples_in_ball.shape[0], samples_in_ball.shape[1], -1)
    local_preds = linear_classifier(samples_in_ball, saliency_map)
    local_detect_of_top_label = (local_preds >=  pred_threshold).float()
    model_detect_of_top_label = (torch.argmax(model_preds, dim=-1) == torch.tensor(top_labels)[:, None]).float()
    accuracies_per_dp = (local_detect_of_top_label == model_detect_of_top_label).float().mean(dim=-1)
    return n_closest, accuracies_per_dp, R


def compute_gradmethod_fidelity_per_kNN(tst_feat, 
                                        imgs,
                                        predictions,
                                        predictions_baseline, 
                                        analysis_data, 
                                        saliency_map, 
                                        predict_fn, 
                                        n_closest, 
                                        tree,
                                        top_labels,
                                        pred_threshold=None):
    if tst_feat.ndim == 1:
        tst_set = tst_set.reshape(1, -1)

    dist, idx= tree.query(tst_feat, k=n_closest, return_distance=True)
    R = np.max(dist, axis=-1)

    # 1. Get all the kNN samples from the analysis dataset
    samples_in_ball = [[analysis_data[idx][0] for idx in row] for row in idx]
    samples_in_ball = torch.stack([torch.stack(row, dim=0) for row in samples_in_ball], dim=0)    
    samples_reshaped = samples_in_ball.reshape(-1, *list(samples_in_ball.shape[2:])) # (num test samples * num closest points) x num features
    with torch.no_grad():
        model_preds = predict_fn(samples_reshaped)

    # 2. Predict labels of the kNN samples with the model
    num_classes = model_preds.shape[-1]
    model_preds = model_preds.reshape(samples_in_ball.shape[0], samples_in_ball.shape[1], num_classes) #num test samples x num closest points x num classes
    if model_preds.shape[-1] == 1:
        model_preds_sig = torch.sigmoid(model_preds)
        model_preds_softmaxed = torch.cat([1 - model_preds_sig, model_preds_sig], dim=-1)
        model_preds_top_label = model_preds.squeeze(-1)
    else: 
        model_preds_softmaxed = torch.softmax(model_preds, dim=-1)
        model_preds_top_label = model_preds[torch.arange(len(model_preds)), :, top_labels].squeeze(-1)

    ## i) Get probability for the top label
    model_probs_top_label = model_preds_softmaxed[torch.arange(len(model_preds_softmaxed)), :, top_labels] # (num test samples,)
    variance_prob_pred = torch.var(model_probs_top_label, dim=1) # (num test samples,)
    variance_pred = torch.var(model_preds_top_label, dim=1) # (num test samples,)
    
    ## ii) Get label predictions, is prediction the same as the top label?
    model_predicted_top_label = (torch.argmax(model_preds_softmaxed, dim=-1) == torch.tensor(top_labels)[:, None]).float()
    
    # 3. Predict labels of the kNN samples with the LIME explanation
    ## i)+ii) Get probability for top label, if prob > threshold, predict as top label
    local_preds = linear_classifier(samples_in_ball, saliency_map) 
    if predictions_baseline.shape[-1] == 1:
        local_preds += predictions_baseline
    else:
        local_preds += predictions_baseline[torch.arange(len(top_labels)), top_labels]
    local_pred_sigmoid = torch.sigmoid(local_preds)
    if pred_threshold is None:
        pred_threshold = 0.5
    local_pred_labels = (local_pred_sigmoid >= pred_threshold).cpu().numpy().astype(int)

    
    # 4. Compute metrics for binary and regression tasks
    res_binary_classification = binary_classification_metrics_per_row(model_predicted_top_label.cpu().numpy(), 
                                                                      local_pred_labels, 
                                                                      local_pred_sigmoid.cpu().numpy(), 
                                                                      )
    res_regression = regression_metrics_per_row(model_preds_top_label.cpu().numpy(),
                                                local_preds.cpu().numpy())
    res_regression_proba = regression_metrics_per_row(model_probs_top_label.cpu().numpy(),
                                                local_pred_sigmoid.cpu().numpy())
    
    res_impurity = impurity_metrics_per_row(model_predicted_top_label.cpu().numpy())
    res_impurity = (*res_impurity, variance_pred, variance_prob_pred)

    return n_closest, res_binary_classification, res_regression, res_regression_proba, res_impurity, R


def compute_saliency_maps(explainer, predict_fn, data_loader_tst, transform = None):
    saliency_map = []
    for i, batch in enumerate(data_loader_tst):
        Xs = batch#[0]
        preds = predict_fn(Xs)
        if preds.ndim == 2:
            saliency = explainer.attribute(Xs).float()
        else:
            top_labels = torch.argmax(predict_fn(Xs), dim=1).tolist()
            saliency = explainer.attribute(Xs, target=top_labels).float()
        saliency_map.append(saliency)
        print("computed the first stack of saliency maps")
    return torch.cat(saliency_map, dim=0)