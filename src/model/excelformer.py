import numpy as np
from torch_frame.nn import ExcelFormer
import torch
from src.dataset.tab_data import TabularDataset
from src.model.base import BaseModelHandler
from src.utils.pytorch_frame_utils import tensor_to_tensorframe, PytorchFrameWrapper, load_dataframes, transform_logit_to_class_proba
from torch_frame import stype, TaskType
from torch_frame.data.tensor_frame import TensorFrame
import torch_frame
from functools import partial
import os
from pathlib import Path

class ExcelFormerHandler(BaseModelHandler):
    def load_model(self):
        # Load the saved state
        data_path = self.data_path
        file_name_wo_file_ending = Path(data_path).stem
        data_folder = os.path.dirname(data_path)
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.col_names_dict = torch.load(os.path.join(data_folder, file_name_wo_file_ending +"_col_names_dict.pt"), map_location=torch.device('cpu'))
        self.col_stats = torch.load(os.path.join(data_folder, file_name_wo_file_ending + "_col_stats.pt"), map_location=torch.device('cpu'))
        
        model = ExcelFormer(**checkpoint['best_model_cfg'], out_channels=1,col_names_dict=self.col_names_dict, col_stats=self.col_stats)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = PytorchFrameWrapper(model, self.col_names_dict)
        return model

    def load_data(self):
        trn_feat, val_feat, whole_tst_feat = load_dataframes(self.data_path)
        tst_feat, analysis_feat, tst_dataset, analysis_dataset = self._split_data_in_tst_analysis(whole_tst_feat,
                                                                                                val_feat,
                                                                                                trn_feat)
        return trn_feat, tst_feat, analysis_feat, tst_dataset, analysis_dataset
    
    def transform(self, X):
        return partial(tensor_to_tensorframe, col_names_dict=self.col_names_dict)(X)

    def predict_fn(self, X):
        pred = self.model(X)
        if self.args.method == "lime":
            pred = transform_logit_to_class_proba(pred)
            pred = pred.detach().numpy()
        return pred
   