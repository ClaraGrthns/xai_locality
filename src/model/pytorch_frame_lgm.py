import numpy as np
from torch_frame.typing import TaskType
from torch_frame.gbdt import LightGBM
import torch
from src.model.base import BaseModelHandler
from src.dataset.tab_data import TabularDataset
from torch.utils.data import DataLoader

class PTFrame_LightGBMHandler(BaseModelHandler):
    def load_model(self):
        model = LightGBM(task_type=TaskType.BINARY_CLASSIFICATION, num_classes=2)
        model.load(self.model_path)
        return model

    def load_data(self):
        data = torch.load(self.data_path)
        train_tensor_frame, val_tensor_frame, test_tensor_frame = data["train"], data["val"], data["test"]
        tst_feat, _ , _ = self.model._to_lightgbm_input(test_tensor_frame)
        val_feat, _, _ = self.model._to_lightgbm_input(val_tensor_frame)
        trn_feat, _, _ = self.model._to_lightgbm_input(train_tensor_frame)
        tst_feat = np.array(tst_feat)
        val_feat = np.array(val_feat)
        trn_feat = np.array(trn_feat)
        tst_feat, analysis_feat, tst_dataset, analysis_dataset = self._split_data_in_tst_analysis(tst_feat,
                                                                                                val_feat,
                                                                                                trn_feat)
        return trn_feat, tst_feat, analysis_feat, tst_dataset, analysis_dataset

    def predict_fn(self, X):
        pred = self.model.model.predict(X)
        if self.model.task_type == TaskType.BINARY_CLASSIFICATION:
            pred = np.column_stack((1 - pred, pred))
        return pred