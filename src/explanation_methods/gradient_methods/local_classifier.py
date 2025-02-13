import numpy as np
from sklearn.neighbors import BallTree
import torch
import torch.nn.functional as F
def linear_classifier(samples_in_ball, saliency_map):
    return torch.einsum('bcij, bkcij -> bk', saliency_map.float(), samples_in_ball)

def linear_approximation(samples_in_ball, reference_img, saliency_map):
    return torch.einsum('bcij, bkcij -> bk', saliency_map.float(), (samples_in_ball - reference_img.unsqueeze(1)))


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


def compute_gradmethod_mse_per_fraction(tst_feat, 
                                        imgs,
                                        predictions,
                                        predictions_baseline, 
                                        analysis_data, 
                                        saliency_map, 
                                        predict_fn, 
                                        n_closest, 
                                        tree):
    if tst_feat.ndim == 1:
        tst_set = tst_set.reshape(1, -1)

    dist, idx= tree.query(tst_feat, k=n_closest, return_distance=True)
    R = np.max(dist, axis=-1)

    samples_in_ball = [[analysis_data[idx][0] for idx in row] for row in idx]
    samples_in_ball = torch.stack([torch.stack(row, dim=0) for row in samples_in_ball], dim=0)    
    samples_reshaped = samples_in_ball.reshape(-1, samples_in_ball.shape[2], samples_in_ball.shape[3], samples_in_ball.shape[4])
    with torch.no_grad():
        model_preds = predict_fn(samples_reshaped)
    model_preds = model_preds.reshape(samples_in_ball.shape[0], samples_in_ball.shape[1], -1) #num test samples x num closest points x num classes
    variance_pred = torch.var(model_preds, dim=1).squeeze(-1) # num test samples x num classes 
    local_preds = linear_classifier(samples_in_ball, saliency_map) + predictions_baseline # num test samples x num closest points x num classes
    local_approx = linear_approximation(samples_in_ball, imgs, saliency_map) + predictions
    if local_preds.ndim == 2:
        local_preds = local_preds.unsqueeze(-1)
    if local_approx.ndim == 2:
        local_approx = local_approx.unsqueeze(-1)
    mse = F.mse_loss(local_preds, model_preds, reduction='none').mean(dim=[1, 2])
    mse_lin_approx = F.mse_loss(local_approx, model_preds, reduction='none').mean(dim=[1, 2])
    return n_closest, mse, mse_lin_approx, variance_pred, R


def compute_saliency_maps(explainer, predict_fn, data_loader_tst, device):
    saliency_map = []
    for i, (imgs, _, _) in enumerate(data_loader_tst):
        preds = predict_fn(imgs)
        if preds.ndim == 1:
            saliency = explainer.attribute(imgs).float()
        else:
            top_labels = torch.argmax(predict_fn(imgs), dim=1).tolist()
            saliency = explainer.attribute(imgs, target=top_labels).float()
        saliency_map.append(saliency)
        print("computed the first stack of saliency maps")
    return torch.cat(saliency_map, dim=0)