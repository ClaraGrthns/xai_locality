import lightgbm as lgb
from src.model.base import BaseModelHandler
import numpy as np

class LightGBMHandler(BaseModelHandler):
    def load_model(self, model_path):
        return lgb.Booster(model_file=model_path)

    def load_data(self, data_path):
        data = np.load(data_path)
        return data['X_test'], data['y_test'], data['X_val'], data['y_val'], data['X_train'], data['y_train']

    def predict_fn(self, X):
        pred = self.model.predict(X)
        if pred.ndim == 1:
            pred = np.column_stack((1 - pred, pred))
        return pred

