import torch
from torch_frame import stype, TaskType
from torch_frame.data.tensor_frame import TensorFrame
import torch_frame

def load_dataframes(data_path):
    data = torch.load(data_path)
    train_tensor_frame, val_tensor_frame, test_tensor_frame = data["train"], data["val"], data["test"]
    trn_feat = tensorframe_to_tensor(train_tensor_frame)
    val_feat = tensorframe_to_tensor(val_tensor_frame)
    tst_feat = tensorframe_to_tensor(test_tensor_frame)
    return trn_feat, val_feat, tst_feat

def tensorframe_to_tensor(tf):
    tst_feat = tf.feat_dict
    dfs = []
    if stype.categorical in tst_feat:
        feat_tensor = tst_feat[stype.categorical]
        dfs.append(feat_tensor)

    if stype.numerical in tst_feat:
        feat_tensor = tst_feat[stype.numerical]
        dfs.append(feat_tensor)

    if stype.embedding in tst_feat:
        feat = tst_feat[stype.embedding]
        feat = feat.values
        feat = feat.view(feat.size(0), -1)
        dfs.append(feat)
    return torch.cat(dfs, dim=1)

 
def tensor_to_tensorframe(tensor, col_names_dict):
    return TensorFrame({torch_frame.numerical: tensor}, col_names_dict) 


class PytorchFrameWrapper(torch.nn.Module):
    def __init__(self, original_model, col_names_dict):
        super(PytorchFrameWrapper, self).__init__()
        self.original_model = original_model
        self.col_names_dict = col_names_dict

    def forward(self, input_tensor):
        tensor_frame = self._convert_to_tensor_frame(input_tensor)
        return self.original_model(tensor_frame)

    def _convert_to_tensor_frame(self, input_tensor):
        return tensor_to_tensorframe(input_tensor, self.col_names_dict)