# model/inception_handler.py
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch import nn
from src.model.base import BaseModelHandler
from src.dataset.imgnet import ImageNetDataset
from src.dataset.cats_vs_dogs import CatsVsDogsDataset
from torchvision import models
from torch.utils.data import DataLoader, Subset
VALIDATION_PATH = "/common/datasets/ImageNet_ILSVRC2012/val"
CLASS_MAPPING_FILE = "/common/datasets/ImageNet_ILSVRC2012/synset_words.txt"

class InceptionV3_Handler(BaseModelHandler):
    def load_model(self):
        """Load a pre-trained InceptionV3 model and modify it to accept feature vectors."""
        device = torch.device("cpu")
        model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        model.eval()
        model.to(device)
        return model

    def load_data(self):
        """Load pre-extracted feature vectors for ImageNet validation and training sets."""
        data_path = VALIDATION_PATH
        specific_classes= self.args.specific_classes
        dataset =  ImageNetDataset(
                data_path, CLASS_MAPPING_FILE, specific_classes=specific_classes, transform="default")
        print("dataset of length: ", len(dataset))
        return dataset
    

    # indices = np.random.permutation(len(tst_feat))
    #     tst_indices, analysis_indices = np.split(indices, [self.args.max_test_points])
    #     print("using the following indices for testing: ", tst_indices)
    #     tst_feat = tst_feat[tst_indices]
    #     df_feat = tst_feat[analysis_indices]
    #     if self.args.include_trn:
    #         df_feat = np.concatenate([trn_feat, df_feat], axis=0)
    #     if self.args.include_val:
    #         df_feat = np.concatenate([df_feat, val_feat], axis=0)  
    #     tst_data = TabularDataset(tst_feat)
    #     analysis_data = TabularDataset(df_feat)

    #     print("Length of data set for analysis", len(analysis_data))
    #     print("Length of test set", len(tst_data))
    #     data_loader_tst = DataLoader(tst_data, batch_size=self.args.chunk_size, shuffle=False)
        
    #     return trn_feat, tst_feat, df_feat, data_loader_tst, df_feat

    
    # def predict_fn(self, X):
    #     """Perform inference using the feature-to-logits model."""
    #     logits = self.model(X)
    #     return F.softmax(logits, dim=1)
    def predict_fn(self, X):
        return self.model(X)

    def get_class_names(self):
        """Load ImageNet class names from the dataset."""
        dataset = ImageNetDataset(
            VALIDATION_PATH, CLASS_MAPPING_FILE, transform="default"
        )
        return dataset.class_names
    
class BinaryInceptionV3_Handler(BaseModelHandler):
    def load_model(self):
        device = torch.device("cpu")
        model = models.inception_v3(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)
        state_dict = torch.load("/home/grotehans/xai_locality/pretrained_models/inception_v3/binary_cat_dog_best.pth", 
                                map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        return model
    
    def load_data(self):
        data_path = "/home/grotehans/xai_locality/data/cats_vs_dogs/test"
        data_path_train = "/home/grotehans/xai_locality/data/cats_vs_dogs/train"
        dataset_tst = CatsVsDogsDataset(data_path, transform="default")
        dataset_trn = CatsVsDogsDataset(data_path_train, transform="default")

        indices = np.random.permutation(len(dataset_tst))
        tst_indices, analysis_indices = np.split(indices, [self.args.max_test_points])
        print("using the following indices for testing: ", tst_indices)

        tst_data = Subset(dataset_tst, tst_indices)
        df_for_expl = Subset(dataset_tst, analysis_indices)
        print("Length of data set for analysis", len(df_for_expl))
        print("Length of test set", len(tst_data))
        
        df_feat = self.load_feature_vectors(path = "/home/grotehans/xai_locality/data/cats_vs_dogs/feature_vectors_binary_inception_cats_dogs_test.csv")
        tst_for_dist, df_for_dist = np.split(df_feat[indices], [self.args.max_test_points])
        # tst_loader_for_expl = DataLoader(tst_data, batch_size=self.args.chunk_size, shuffle=False)

        return dataset_trn, tst_for_dist, df_for_dist, tst_data, df_for_expl
    
    # def predict_fn(self, X):
    #     preds = torch.sigmoid(self.model(X))
    #     return torch.cat([1 - preds, preds], dim=1)

    def predict_fn(self, X):
        return self.model(X)

    def get_class_names(self):
        return ["Cat", "Dog"]
    
    def load_feature_vectors(self, path):
        """Load pre-extracted feature vectors"""
        feat_vec = pd.read_csv(path)
        feat_vec = feat_vec.drop(columns=['label'])
        feat_vec = feat_vec.drop(columns=['path'])
        return feat_vec.to_numpy()

class InceptionV3BinaryClassifier(nn.Module):
    def __init__(self, pretrained=True, device="cpu"):
        super(InceptionV3BinaryClassifier, self).__init__()
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: 
            device = torch.device(device)
        self.model = models.inception_v3(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, 1)
        state_dict = torch.load("/home/grotehans/xai_locality/pretrained_models/inception_v3/binary_cat_dog_best.pth", 
                        map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        
    def forward(self, x):
        preds = torch.sigmoid(self.model(x))
        return torch.cat([1 - preds, preds], dim=1)