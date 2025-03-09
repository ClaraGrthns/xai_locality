
# model/base.py
import numpy as np
from src.dataset.tab_data import TabularDataset
from src.utils.misc import get_path
import torch
from src.utils.pytorch_frame_utils import (
   tensor_to_tensorframe,
    tensorframe_to_tensor
) 
class BaseModelHandler:
    def __init__(self, args):
        self.args = args
        if (args.data_path or args.data_folder) is not None:
            self.data_path = self.get_data_path()
            print(f"Loading data from: {self.data_path}")
        if (args.model_path or args.model_folder) is not None:
            self.model_path = self.get_model_path()
            print(f"Loading model from: {self.model_path}")
        self.model = self.load_model()

    def get_model_path(self):
        model_path = get_path(self.args.model_folder, self.args.model_path, self.args.setting, suffix=self.args.model_type + "_")
        return model_path
    
    def get_data_path(self):
        data_path = get_path(self.args.data_folder, self.args.data_path, self.args.setting)
        if self.args.setting is not None:
            data_path += ".npz"
        return data_path
    
    def _get_split_indices(self, whole_tst_feat):
        indices = np.random.permutation(len(whole_tst_feat))
        tst_indices, analysis_indices = np.split(indices, [self.args.max_test_points])
        print("using the following indices for testing: ", tst_indices)
        return tst_indices, analysis_indices
    
    def _get_tst_feat_label_forKNN(self, whole_tst_feat, y):
        tst_indices, analysis_indices = self._get_split_indices(whole_tst_feat)
        analysis_feat = whole_tst_feat[analysis_indices]
        tst_feat = whole_tst_feat[tst_indices]
        analysis_y = y[analysis_indices]
        tst_y = y[tst_indices]
        return tst_feat, analysis_feat, tst_y, analysis_y
    
    def load_data_for_kNN(self):
        if self.data_path.endswith(".pt"):
            data = torch.load(self.data_path)
            test_tensor_frame = data["test"]
            whole_tst_feat = tensorframe_to_tensor(test_tensor_frame).numpy()
            y = test_tensor_frame.y.numpy()
            trn_tensor_frame = data["train"]
            trn_feat = tensorframe_to_tensor(trn_tensor_frame).numpy()
            y_trn = trn_tensor_frame.y.numpy()
        else:
            data = np.load(self.data_path)
            whole_tst_feat = data['X_test']
            y = data['y_test']
            trn_feat = data['X_train']
            y_trn = data['y_train']
        tst_feat, analysis_feat, tst_y, analysis_y = self._get_tst_feat_label_forKNN(whole_tst_feat, y)
        return trn_feat, analysis_feat, tst_feat, y_trn, analysis_y, tst_y 
    
    def _split_data_in_tst_analysis(self, whole_tst_feat, val_feat, trn_feat):
        tst_indices, analysis_indices = self._get_split_indices(whole_tst_feat)
        analysis_feat = whole_tst_feat[analysis_indices]
        tst_feat = whole_tst_feat[tst_indices]
        if self.args.include_trn:
            if isinstance(trn_feat, np.ndarray):
                analysis_feat = np.concatenate([analysis_feat, trn_feat], axis=0)
            else:
                analysis_feat = torch.cat([analysis_feat, trn_feat], dim=0)
        if self.args.include_val:
            if isinstance(val_feat, np.ndarray):
                analysis_feat = np.concatenate([analysis_feat, val_feat], axis=0)
            else:
                analysis_feat = torch.cat([analysis_feat, val_feat], dim=0) 
        tst_dataset = TabularDataset(tst_feat)
        analysis_dataset = TabularDataset(analysis_feat)
        print("Length of data set for analysis", len(analysis_dataset))
        print("Length of test set", len(tst_dataset)) 
        return tst_feat, analysis_feat, tst_dataset, analysis_dataset

    def load_model(self):
        """Load model from path"""
        raise NotImplementedError
    
    def load_feature_vectors(self):
        return None

    def load_data(self):
        """Load dataset"""
        raise NotImplementedError

    def predict_fn(self, X):
        """Run predictions"""
        raise NotImplementedError

    def get_feature_names(self, trn_feat):
        """Return feature names (if applicable)"""
        return np.arange(trn_feat.shape[1])  # Default behavior
    
    def get_class_names(self):
        """Return class names (if applicable)"""
        return np.arange(2)
