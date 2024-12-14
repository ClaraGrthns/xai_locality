import xgboost
import numpy as np
from torch_frame.typing import TaskType
from torch_frame.gbdt import XGBoost
import torch


def load_model(model_path):
    model = XGBoost(task_type=TaskType.BINARY_CLASSIFICATION, num_classes=2)
    model.load(model_path)
    return model

def load_data(model, data_path):
    data = torch.load(data_path)
    train_tensor_frame, val_tensor_frame, test_tensor_frame = data["train"], data["val"], data["test"]
    tst_feat, tst_y, _ = model._to_xgboost_input(test_tensor_frame)
    val_feat, val_y, _ = model._to_xgboost_input(val_tensor_frame)
    trn_feat, trn_y, _ = model._to_xgboost_input(train_tensor_frame)
    return tst_feat, tst_y, val_feat, val_y, trn_feat, trn_y

def predict_fn(X, model):
    types = ["q"] * X.shape[1]
    if X.ndim == 1:
        X = X.reshape(1, -1)
    dummy_labels = np.zeros(X.shape[0])
    dtest = xgboost.DMatrix(X, label=dummy_labels,
                            feature_types=types,
                            enable_categorical=True)
    pred = model.model.predict(dtest)
    if model.task_type == TaskType.BINARY_CLASSIFICATION:
        pred = np.column_stack((1 - pred, pred))
    return pred
def get_feature_names(tensor_frame):
    first_key = next(iter(tensor_frame.col_names_dict))
    feature_names = tensor_frame.col_names_dict[first_key]
    return feature_names