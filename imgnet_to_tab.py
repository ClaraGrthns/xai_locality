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



# Downsample function
def downsample_dataset(dataset, fraction_per_class=0.1):
    """
    Downsamples the dataset by selecting a fraction of samples per class.
    
    Args:
        dataset: PyTorch Dataset (e.g., ImageNet training dataset).
        fraction_per_class: Fraction of images to keep per class (default is 0.1 or 10%).
        
    Returns:
        Subset of the original dataset with reduced size.
    """
    class_indices = defaultdict(list)
    for idx, (_,_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    downsampled_indices = []
    for label, indices in class_indices.items():
        num_samples = max(1, int(len(indices) * fraction_per_class))  # Ensure at least 1 sample per class
        downsampled_indices.extend(random.sample(indices, num_samples))
    return Subset(dataset, downsampled_indices)

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
OUTPUT_CSV_TRN = "/home/grotehans/xai_locality/data/feature_vectors_img_net_trn_downsampled.csv"


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
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
features, labels, paths = extract_features(dataloader, model)
feature_df = pd.DataFrame(features)
feature_df['label'] = labels
feature_df['image_path'] = paths

feature_df.to_csv(OUTPUT_CSV, index=False)

print(f"Feature extraction completed and saved to {OUTPUT_CSV}")

# Process downsampled validation dataset
downsampled_validation_dataset = downsample_dataset(dataset, fraction_per_class=0.2)
dataloader_downsampled = DataLoader(downsampled_validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
features_downsampled, labels_downsampled, paths_downsampled = extract_features(dataloader_downsampled, model)
feature_df_downsampled = pd.DataFrame(features_downsampled)
feature_df_downsampled['label'] = labels_downsampled
feature_df_downsampled['image_path'] = paths_downsampled

OUTPUT_CSV_VAL_DOWNSAMPLED = "/home/grotehans/xai_locality/data/feature_vectors_img_net_val_downsampled.csv"
feature_df_downsampled.to_csv(OUTPUT_CSV_VAL_DOWNSAMPLED, index=False)

print(f"Downsampled feature extraction completed and saved to {OUTPUT_CSV_VAL_DOWNSAMPLED}")

training_dataset = ImageNetValidationDataset(
    TRAINING_PATH, CLASS_MAPPING_FILE, transform=get_default_transforms()
)

downsampled_training_dataset = downsample_dataset(training_dataset, fraction_per_class=0.1)
dataloader = DataLoader(downsampled_training_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
features, labels, paths = extract_features(dataloader, model)
feature_df = pd.DataFrame(features)
feature_df['label'] = labels
feature_df['image_path'] = paths

feature_df.to_csv(OUTPUT_CSV_TRN, index=False)

print(f"Feature extraction completed and saved to {OUTPUT_CSV_TRN}")



