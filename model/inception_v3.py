import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
VALIDATION_PATH = "/common/datasets/ImageNet_ILSVRC2012/val"
CLASS_MAPPING_FILE = "/common/datasets/ImageNet_ILSVRC2012/synset_words.txt"
BATCH_SIZE = 64 
OUTPUT_CSV = "/home/grotehans/xai_locality/data/feature_vectors_img_net_val.csv"
from data import ImageNetValidationDataset


def predict_fn(X, model):
    X = torch.tensor(X).float()
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()

def load_model(model_path):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.eval()
    # Define a new model to handle feature vectors as input and produce logits
    class FeatureToLogitsModel(nn.Module):
        def __init__(self, original_model):
            super(FeatureToLogitsModel, self).__init__()
            self.fc = original_model.fc  # Copy the last linear layer

        def forward(self, x):
            return self.fc(x)

    # Create the new model
    feature_to_logits_model = FeatureToLogitsModel(model)
    return feature_to_logits_model

def load_data(model, data_path):
    file_path_val = "/home/grotehans/xai_locality/data/feature_vectors_img_net_val.csv"
    file_path_trn = "/home/grotehans/xai_locality/data/feature_vectors_img_net_trn_downsampled_5.0perc.csv"
    df_val = pd.read_csv(file_path_val)
    X_test = df_val.drop(columns=['label', 'image_path']).values
    y_test = df_val['label'].values
    # X_trn, X_test, y_trn, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
    df_trn = pd.read_csv(file_path_trn)
    X_trn = df_trn.drop(columns=['label', 'image_path']).values
    y_trn = df_trn['label'].values
    return X_test, y_test, np.empty(0),  np.empty(0),  X_trn, y_trn

def get_feature_names(trn_feat):
    return np.arange(trn_feat.shape[1])

def get_class_names():
    dataset = ImageNetValidationDataset(
    VALIDATION_PATH, CLASS_MAPPING_FILE, transform="default"
    )
    return dataset.class_names
