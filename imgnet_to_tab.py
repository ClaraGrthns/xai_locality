from torchvision import transforms
from data import ImageNetValidationDataset
import torch
import torch.nn.functional as F
from utils.preprocessing import get_preprocess_transform, get_pil_transform, grid_wrapper
import pandas as pd
import random
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from torch.utils.data import Subset

def get_default_transforms():
    """Get the default ImageNet transforms."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
VALIDATION_PATH = "/common/datasets/ImageNet_ILSVRC2012/val"
TRAINING_PATH = "/common/datasets/ImageNet_ILSVRC2012/train"
CLASS_MAPPING_FILE = "/common/datasets/ImageNet_ILSVRC2012/synset_words.txt"
BATCH_SIZE = 64 
OUTPUT_CSV = "/home/grotehans/xai_locality/data/feature_vectors_img_net_val.csv"


model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.eval()
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.avgpool.register_forward_hook(get_activation("avgpool"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def extract_features(dataloader, model):
    all_features = []
    all_labels = []
    all_paths = []

    with torch.no_grad(): 
        for batch in dataloader:
            imgs, labels, paths = batch  
            imgs = imgs.to(device) 
            _ = model(imgs)  
            features = activation['avgpool'].squeeze().cpu().numpy()
            all_features.append(features)
            all_labels.extend(labels)
            all_paths.extend(paths)

    all_features = np.vstack(all_features)
    return all_features, all_labels, all_paths

# Process validation dataset
dataset = ImageNetValidationDataset(
    VALIDATION_PATH, CLASS_MAPPING_FILE, transform=get_default_transforms()
)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
# features, labels, paths = extract_features(dataloader, model)
# feature_df = pd.DataFrame(features)
# feature_df['label'] = labels
# feature_df['image_path'] = paths

# feature_df.to_csv(OUTPUT_CSV, index=False)

# print(f"Feature extraction completed and saved to {OUTPUT_CSV}")

fraction_per_class = 0.1
OUTPUT_CSV_TRN = f"/home/grotehans/xai_locality/data/feature_vectors_img_net_trn_downsampled_{fraction_per_class*100}perc.csv"

training_dataset = ImageNetValidationDataset(
    TRAINING_PATH, CLASS_MAPPING_FILE, transform=get_default_transforms(), fraction_per_class=fraction_per_class
)

print("initalised downsampled training dataset")

dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
features, labels, paths = extract_features(dataloader, model)
feature_df = pd.DataFrame(features)
feature_df['label'] = labels
feature_df['image_path'] = paths

feature_df.to_csv(OUTPUT_CSV_TRN, index=False)

print(f"Feature extraction completed and saved to {OUTPUT_CSV_TRN}")



