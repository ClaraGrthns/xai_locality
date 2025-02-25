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
        print(trn_feat.shape)

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
        
        return trn_feat, tst_feat, df_feat, tst_data, analysis_data


    def predict_fn(self, X):
        pred = self.model.model.predict(X)
        if self.model.task_type == TaskType.BINARY_CLASSIFICATION:
            pred = np.column_stack((1 - pred, pred))
        return pred