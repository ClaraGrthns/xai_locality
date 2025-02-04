# model/xgboost_handler.py
import numpy as np
import torch
import xgboost
from torch_frame.typing import TaskType
from torch_frame.gbdt import XGBoost
from model.base import BaseModelHandler

class PTFrame_XGBoostHandler(BaseModelHandler):
    def load_model(self, model_path):
        """Load an XGBoost model using the TorchFrame wrapper."""
        model = XGBoost(task_type=TaskType.BINARY_CLASSIFICATION, num_classes=2)
        model.load(model_path)
        return model

    def load_data(self, data_path):
        """Load train, validation, and test datasets from a Torch tensor frame."""
        data = torch.load(data_path)
        train_tensor_frame, val_tensor_frame, test_tensor_frame = data["train"], data["val"], data["test"]

        tst_feat, tst_y, _ = self.model._to_xgboost_input(test_tensor_frame)
        val_feat, val_y, _ = self.model._to_xgboost_input(val_tensor_frame)
        trn_feat, trn_y, _ = self.model._to_xgboost_input(train_tensor_frame)

        return tst_feat, tst_y, val_feat, val_y, trn_feat, trn_y

    def predict_fn(self, X):
        """Perform inference using the XGBoost model."""
        types = ["q"] * X.shape[1]
        if X.ndim == 1:
            X = X.reshape(1, -1)

        dummy_labels = np.zeros(X.shape[0])
        dtest = xgboost.DMatrix(X, label=dummy_labels, feature_types=types, enable_categorical=True)
        pred = self.model.model.predict(dtest)

        if self.model.task_type == TaskType.BINARY_CLASSIFICATION:
            pred = np.column_stack((1 - pred, pred))
        return pred

    def get_feature_names(self, tensor_frame):
        """Extract feature names from the Torch tensor frame."""
        first_key = next(iter(tensor_frame.col_names_dict))
        return tensor_frame.col_names_dict[first_key]

