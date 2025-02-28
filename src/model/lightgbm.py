import lightgbm as lgb
from src.model.base import BaseModelHandler
import numpy as np
from src.utils.misc import get_path
from src.dataset.tab_data import TabularDataset
from torch.utils.data import DataLoader

class LightGBMHandler(BaseModelHandler):
    def load_model(self):
        model_path = self.model_path
        return lgb.Booster(model_file=model_path)

    def load_data(self):
        data_path = self.data_path
        data = np.load(data_path)
        whole_tst_feat, val_feat, trn_feat = data['X_test'], data['X_val'], data['X_train']
        tst_feat, analysis_feat, tst_dataset, analysis_dataset = self._split_data_in_tst_analysis(whole_tst_feat, 
                                                                                                  val_feat, 
                                                                                                  trn_feat)
        return trn_feat, tst_feat, analysis_feat, tst_dataset, analysis_dataset

    def predict_fn(self, X):
        pred = self.model.predict(X)
        if pred.ndim == 1:
            pred = np.column_stack((1 - pred, pred))
        return pred
