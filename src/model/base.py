
# model/base.py
import numpy as np

class BaseModelHandler:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Load model from path"""
        raise NotImplementedError

    def load_data(self, data_path=None):
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
