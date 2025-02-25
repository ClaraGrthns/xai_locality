import numpy as np
from torch_frame.nn import ExcelFormer
import torch
from src.dataset.tab_data import TabularDataset
from src.model.base import BaseModelHandler
from src.utils.pytorch_frame_utils import tensorframe_to_tensor, tensor_to_tensorframe, PytorchFrameWrapper
from torch_frame import stype, TaskType
from torch_frame.data.tensor_frame import TensorFrame
import torch_frame
from functools import partial




class ExcelFormerHandler(BaseModelHandler):
    def load_model(self):
        # Load the saved state
        checkpoint = torch.load("/home/grotehans/xai_locality/pretrained_models/excelformer/ExcelFormer_normalized_binary_large_0.pt", map_location=torch.device('cpu'))
        self.col_names_dict = torch.load("/home/grotehans/xai_locality/data/col_names_dict.pt", map_location=torch.device('cpu'))
        self.col_stats = torch.load("/home/grotehans/xai_locality/data/col_stats.pt", map_location=torch.device('cpu'))
        model = ExcelFormer(**checkpoint['best_model_cfg'], out_channels=1,col_names_dict=self.col_names_dict, col_stats=self.col_stats)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = PytorchFrameWrapper(model, self.col_names_dict)
        return model

    def load_data(self):
        data = torch.load(self.data_path)
        train_tensor_frame, val_tensor_frame, test_tensor_frame = data["train"], data["val"], data["test"]
        self.col_names_dict = train_tensor_frame.col_names_dict
        trn_feat = tensorframe_to_tensor(train_tensor_frame)
        val_feat = tensorframe_to_tensor(val_tensor_frame)
        tst_feat = tensorframe_to_tensor(test_tensor_frame)
        print(type(tst_feat))
        indices = np.random.permutation(len(tst_feat))
        tst_indices, analysis_indices = np.split(indices, [self.args.max_test_points])
        print("using the following indices for testing: ", tst_indices)
        df_feat = tst_feat[analysis_indices]
        tst_feat = tst_feat[tst_indices]
        if self.args.include_trn:
            df_feat = torch.concat([trn_feat, df_feat], dim=0)
        if self.args.include_val:
            df_feat = torch.concat([df_feat, val_feat], dim=0)  
        tst_data = TabularDataset(tst_feat)
        analysis_data = TabularDataset(df_feat)
        print("Length of data set for analysis", len(analysis_data))
        print("Length of test set", len(tst_data))
        return trn_feat, tst_feat, df_feat, tst_data, analysis_data
    
    def transform(self, X):
        return partial(tensor_to_tensorframe, col_names_dict=self.col_names_dict)(X)

    def predict_fn(self, X):
        if type(X) == np.array:
            X = torch.tensor(X)
        pred = self.model(X)
        # sigmoid predictions
        # pred = torch.sigmoid(pred)
        # pred = pred.detach().numpy()
        # if self.model.task_type == TaskType.BINARY_CLASSIFICATION:
        #     pred = np.column_stack((1 - pred, pred))
        return pred
   