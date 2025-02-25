
# model/base.py
import numpy as np
from src.utils.misc import get_path

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
        model_path = get_path(self.args.model_folder, self.args.model_path, self.args.setting, suffix="final_model_")
        return model_path
    
    def get_data_path(self):
        data_path = get_path(self.args.data_folder, self.args.data_path, self.args.setting)
        if self.args.setting is not None:
            data_path += ".npz"
        return data_path

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
