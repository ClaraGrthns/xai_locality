import os
from pathlib import Path
import torch
from functools import partial
from src.model.base import BaseModelHandler
from src.utils.pytorch_frame_utils import (
    tensor_to_tensorframe, 
    PytorchFrameWrapper, 
    load_dataframes, 
    tensorframe_to_tensor,
    transform_logit_to_class_proba
)
import numpy as np

class TorchFrameHandler(BaseModelHandler):
    def __init__(self, args, model_class):
        self.model_class = model_class
        super().__init__(args)

    def load_model(self):
        if self.model_class is None:
            raise ValueError("Model class must be provided to TorchFrameHandler")

        data_path = self.data_path
        file_name_wo_file_ending = Path(data_path).stem
        data_folder = os.path.dirname(data_path)
        
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.col_names_dict = torch.load(
            os.path.join(data_folder, file_name_wo_file_ending + "_col_names_dict.pt"), 
            map_location=torch.device('cpu')
        )
        self.col_stats = torch.load(
            os.path.join(data_folder, file_name_wo_file_ending + "_col_stats.pt"), 
            map_location=torch.device('cpu')
        )

        model = self.model_class(
            **checkpoint['best_model_cfg'], 
            out_channels=1,
            col_names_dict=self.col_names_dict, 
            col_stats=self.col_stats
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = PytorchFrameWrapper(model, self.col_names_dict)
        return model

    def load_data(self):
        trn_feat, val_feat, whole_tst_feat = load_dataframes(self.data_path)
        tst_feat, analysis_feat, tst_dataset, analysis_dataset = self._split_data_in_tst_analysis(
            whole_tst_feat, val_feat, trn_feat
        )
        return trn_feat, tst_feat, analysis_feat, tst_dataset, analysis_dataset
    
    def transform(self, X):
        return partial(tensor_to_tensorframe, col_names_dict=self.col_names_dict)(X)

    def predict_fn(self, X):
        if isinstance(X, np.ndarray):
            X = torch.Tensor(X)
        if X.dtype == torch.double:
            X = X.float()
        pred = self.model(X)
        if self.args.method == "lime":
            pred = transform_logit_to_class_proba(pred)
            pred = pred.detach().numpy()
        return pred
