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
        tst_feat, val_feat, trn_feat = data['X_test'], data['X_val'], data['X_train']
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
        pred = self.model.predict(X)
        if pred.ndim == 1:
            pred = np.column_stack((1 - pred, pred))
        return pred

# data_path = get_path(args.data_folder, args.data_path, args.setting)
#     model_path = get_path(args.model_folder, args.model_path, args.setting, suffix="final_model_")
#     results_path = get_path(args.results_folder, args.results_path, args.setting)