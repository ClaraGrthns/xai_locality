
# model/base.py
import numpy as np
from src.utils.misc import get_path

class BaseModelHandler:
    def __init__(self, args):
        self.args = args
        if args.method =="lime":
            self.data_path = self.get_data_path()
            self.model_path = self.get_model_path()
        self.model = self.load_model()

    def get_model_path(self):
        model_path = get_path(self.args.model_folder, self.args.model_path, self.args.setting)
        return model_path
    
    def get_data_path(self):
        data_path = get_path(self.args.data_folder, self.args.data_path, self.args.setting)
        return data_path

    def load_model(self):
        """Load model from path"""
        raise NotImplementedError

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
