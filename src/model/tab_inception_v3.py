# model/inception_handler.py
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch import nn
from src.model.base import BaseModelHandler
from src.dataset.imgnet import ImageNetDataset

VALIDATION_PATH = "/common/datasets/ImageNet_ILSVRC2012/val"
CLASS_MAPPING_FILE = "/common/datasets/ImageNet_ILSVRC2012/synset_words.txt"

class TabInceptionV3Handler(BaseModelHandler):
    def load_model(self):
        """Load a pre-trained InceptionV3 model and modify it to accept feature vectors."""
        model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        model.eval()

        # Define a wrapper model that only includes the final fully connected layer
        class FeatureToLogitsModel(nn.Module):
            def __init__(self, original_model):
                super(FeatureToLogitsModel, self).__init__()
                self.fc = original_model.fc  # Use only the last fully connected layer

            def forward(self, x):
                return self.fc(x)
            
        return FeatureToLogitsModel(model)

    def load_data(self):
        """Load pre-extracted feature vectors for ImageNet validation and training sets."""
        file_path_val = "/home/grotehans/xai_locality/data/feature_vectors_img_net_val.csv"
        file_path_trn = "/home/grotehans/xai_locality/data/feature_vectors_img_net_trn_downsampled_5.0perc.csv"

        # Load validation data
        df_val = pd.read_csv(file_path_val)
        X_test = df_val.drop(columns=['label', 'image_path']).values
        y_test = df_val['label'].values

        # Load training data
        df_trn = pd.read_csv(file_path_trn)
        X_trn = df_trn.drop(columns=['label', 'image_path']).values
        y_trn = df_trn['label'].values

        return X_test, y_test, np.empty(0), np.empty(0), X_trn, y_trn

    def predict_fn(self, X):
        """Perform inference using the feature-to-logits model."""
        X = torch.tensor(X).float()
        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def get_class_names(self):
        """Load ImageNet class names from the dataset."""
        dataset = ImageNetDataset(
            VALIDATION_PATH, CLASS_MAPPING_FILE, transform="default"
        )
        return dataset.class_names
