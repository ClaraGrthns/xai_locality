import lightgbm as lgb
import numpy as np

def predict_fn(X, model):
    pred = model.predict(X)
    if pred.ndim == 1:
        pred = np.column_stack((1 - pred, pred))
    return pred

def load_model(model_path):
    return lgb.Booster(model_file=model_path)

def load_data(model, data_path):
    data = np.load(data_path)
    tst_feat, tst_y, val_feat, val_y, trn_feat, trn_y = data['X_test'], data['y_test'], data['X_val'], data['y_val'], data['X_train'], data['y_train']
    return tst_feat, tst_y, val_feat, val_y, trn_feat, trn_y

def get_feature_names(trn_feat):
    return np.arange(trn_feat.shape[1])
