# model/inception_handler.py
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch import nn
from model.base import BaseModelHandler
from dataset.imgnet import ImageNetValidationDataset

VALIDATION_PATH = "/common/datasets/ImageNet_ILSVRC2012/val"
CLASS_MAPPING_FILE = "/common/datasets/ImageNet_ILSVRC2012/synset_words.txt"

class InceptionV3_Handler(BaseModelHandler):
    def load_model(self, model_path):
        """Load a pre-trained InceptionV3 model and modify it to accept feature vectors."""
        model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        model.eval()
        return model

    def load_data(self, data_path):
        """Load pre-extracted feature vectors for ImageNet validation and training sets."""
        dataset = ImageNetValidationDataset(
                VALIDATION_PATH, CLASS_MAPPING_FILE)
        return dataset
    
    def predict_fn(self, X):
        """Perform inference using the feature-to-logits model."""
        logits = self.model(X)
        return F.softmax(logits, dim=1)

    def get_class_names(self):
        """Load ImageNet class names from the dataset."""
        dataset = ImageNetValidationDataset(
            VALIDATION_PATH, CLASS_MAPPING_FILE, transform="default"
        )
        return dataset.class_names
