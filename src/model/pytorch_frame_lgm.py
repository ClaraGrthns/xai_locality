import numpy as np
from torch_frame.typing import TaskType
from torch_frame.gbdt import LightGBM
import torch
from src.model.base import BaseModelHandler

class PTFrame_LightGBMHandler(BaseModelHandler):
    def load_model(self, model_path):
        model = LightGBM(task_type=TaskType.BINARY_CLASSIFICATION, num_classes=2)
        model.load(model_path)
        return model

    def load_data(self, data_path):
        data = torch.load(data_path)
        train_tensor_frame, val_tensor_frame, test_tensor_frame = data["train"], data["val"], data["test"]
        tst_feat, tst_y, _ = self.model._to_lightgbm_input(test_tensor_frame)
        val_feat, val_y, _ = self.model._to_lightgbm_input(val_tensor_frame)
        trn_feat, trn_y, _ = self.model._to_lightgbm_input(train_tensor_frame)
        return np.array(tst_feat), np.array(tst_y), np.array(val_feat), np.array(val_y), np.array(trn_feat), np.array(trn_y)

    def predict_fn(self, X):
        pred = self.model.model.predict(X)
        if self.model.task_type == TaskType.BINARY_CLASSIFICATION:
            pred = np.column_stack((1 - pred, pred))
        return pred