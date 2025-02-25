# model/xgboost_handler.py
import numpy as np
import torch
import xgboost
from torch_frame.typing import TaskType
from torch_frame.gbdt import XGBoost
from torch.utils.data import DataLoader
from src.model.base import BaseModelHandler
from src.dataset.tab_data import TabularDataset

class PTFrame_XGBoostHandler(BaseModelHandler):
    def load_model(self):
        """Load an XGBoost model using the TorchFrame wrapper."""
        model = XGBoost(task_type=TaskType.BINARY_CLASSIFICATION, num_classes=2)
        model.load(self.model_path)
        return model

    def load_data(self):
        """Load train, validation, and test datasets from a Torch tensor frame."""
        data = torch.load(self.data_path)
        train_tensor_frame, val_tensor_frame, test_tensor_frame = data["train"], data["val"], data["test"]

        tst_feat, tst_y, _ = self.model._to_xgboost_input(test_tensor_frame)
        val_feat, val_y, _ = self.model._to_xgboost_input(val_tensor_frame)
        trn_feat, trn_y, _ = self.model._to_xgboost_input(train_tensor_frame)
        indices = np.random.permutation(len(tst_feat))
        tst_indices, analysis_indices = np.split(indices, [self.args.max_test_points])
        print("using the following indices for testing: ", tst_indices)
        df_feat = tst_feat[analysis_indices]
        tst_feat = tst_feat[tst_indices]
        if self.args.include_trn:
            df_feat = np.concatenate([trn_feat, df_feat], axis=0)
        if self.args.include_val:
            df_feat = np.concatenate([df_feat, val_feat], axis=0)  
        tst_data = TabularDataset(tst_feat)
        analysis_data = TabularDataset(df_feat)

        print("Length of data set for analysis", len(analysis_data))
        print("Length of test set", len(tst_data))
        # data_loader_tst = DataLoader(tst_data, batch_size=self.args.chunk_size, shuffle=False)
        
        return trn_feat, tst_feat, df_feat, tst_data, df_feat
    
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

