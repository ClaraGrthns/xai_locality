import numpy as np
from sklearn.neighbors import BallTree
import torch
def linear_classifier(samples_in_ball, saliency_map):
    return torch.einsum('bcij, bkcij -> bk', saliency_map.float(), samples_in_ball)

def compute_gradmethod_accuracy_per_fraction(tst_feat, top_labels, analysis_data, saliency_map, predict_fn, n_closest, tree, device, pred_threshold=None):
    if tst_feat.ndim == 1:
        tst_set = tst_set.reshape(1, -1)

    dist, idx= tree.query(tst_feat, k=n_closest, return_distance=True)
    R = np.max(dist, axis=-1)

    samples_in_ball = [[analysis_data[idx][0] for idx in row] for row in idx]
    samples_in_ball = torch.stack([torch.stack(row, dim=0) for row in samples_in_ball], dim=0)    
    samples_in_ball = samples_in_ball.to(device)
    samples_reshaped = samples_in_ball.reshape(-1, samples_in_ball.shape[2], samples_in_ball.shape[3], samples_in_ball.shape[4])

    model_preds = predict_fn(samples_reshaped)
    model_preds = model_preds.reshape(samples_in_ball.shape[0], samples_in_ball.shape[1], -1)
    local_preds = linear_classifier(samples_in_ball, saliency_map)
    local_detect_of_top_label = (local_preds >=  pred_threshold).float()
    
    model_detect_of_top_label = (torch.argmax(model_preds, dim=-1) == torch.tensor(top_labels).to(device)[:, None]).float()
    accuracies_per_dp = (local_detect_of_top_label == model_detect_of_top_label).float().mean(dim=-1)
    return n_closest, accuracies_per_dp, R