import os
import random
from typing import List, Tuple, Dict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageNetValidationDataset(Dataset):
    """
    A PyTorch Dataset for loading ImageNet validation data.
    
    Attributes:
        data_list (List[Tuple[str, str]]): List of (image_path, class_name) tuples
        transform: PyTorch transforms to apply to images, if 'default' use default ImageNet transforms
        class_mapping (Dict[str, str]): Mapping from WordNet IDs to class names
        wnids (List[str]): List of WordNet IDs
        class_names (List[str]): List of class names
    """
    
    def __init__(
        self,
        validation_path: str,
        class_mapping_file: str,
        num_classes: int = None,
        transform=None,
        fraction_per_class: float = 1.0,
        seed: int = 42,
        specific_classes: List[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            validation_path (str): Path to ImageNet validation set
            class_mapping_file (str): Path to synset words mapping file
            num_classes (int): Number of classes to sample (default: 200)
            transform: Optional transforms to apply to images
            seed (int): Random seed for class sampling
            specific_classes (List[str]): List of specific classes to include
        """
        if transform == "default":
            transform = get_default_transforms()
        self.transform = transform
        
        self.wnids, self.class_names = self._load_class_mapping(class_mapping_file)
        self.wnid_to_class = dict(zip(self.wnids, self.class_names))
        self.class_to_wnid = dict(zip(self.class_names, self.wnids))

        assert all(cls in self.class_names for cls in specific_classes), "Specific classes not found in ImageNet"        
        self.data_list = self._create_data_list(validation_path, num_classes, seed, fraction_per_class, specific_classes)
    
    def _load_class_mapping(self, mapping_file: str) -> Tuple[List[str], List[str]]:
        """Load mapping between WordNet IDs and class names."""
        wnids = []
        class_names = []
        
        with open(mapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    wnid, class_name = parts
                    class_name = class_name.split(', ')[0].strip()
                    class_names.append(class_name)
                    wnids.append(wnid)
                    
        return wnids, class_names
    
    def _create_data_list(
        self,
        validation_path: str,
        num_classes: int,
        seed: int,
        fraction_per_class: float = 1.0,
        specific_classes: List[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Create list of (image_path, class_name) tuples with optional subsampling of files.
        
        Args:
            validation_path (str): Path to the validation dataset.
            num_classes (int): Number of classes to sample. If None, include all classes.
            seed (int): Random seed for deterministic sampling.
            fraction_per_class (float): Fraction of files to include per class (default: 1.0, i.e., all files).
            specific_classes (List[str]): List of specific classes to include.
        
        Returns:
            List[Tuple[str, str]]: List of (image_path, class_name) tuples.
        """
        random.seed(seed)
        class_names = os.listdir(validation_path)
        
        if specific_classes:
            sampled_classes = [self.class_to_wnid[class_name] for class_name in specific_classes]
        elif num_classes:
            sampled_classes = random.sample(class_names, num_classes)
        else:
            sampled_classes = class_names

        self.class_names = [self.wnid_to_class[wnid] for wnid in sampled_classes]
        self.wnids = sampled_classes
    
        data_list = []
        for class_name in sampled_classes:
            class_path = os.path.join(validation_path, class_name)
            # Get all files in the class directory and subsample
            all_files = os.listdir(class_path)
            if fraction_per_class < 1.0:
                print(f"Subsampling {fraction_per_class} of files from class {class_name}")
                print("Before:", len(all_files), "After:", max(1, int(len(all_files) * fraction_per_class)))
                num_files_to_sample = max(1, int(len(all_files) * fraction_per_class))
                sampled_files = random.sample(all_files, num_files_to_sample)
            else:
                sampled_files = all_files  # Use all files if fraction is 1.0
            # Create (file_path, class_name) tuples
            for file_name in sampled_files:
                file_path = os.path.join(class_path, file_name)
                data_list.append((file_path, class_name))
        return data_list
    
    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        """
        Get an image and its class name.
        
        Args:
            idx (int): Index of the data item
            
        Returns:
            tuple: (image, class_name) where image is the transformed PIL Image
                  and class_name is the human-readable class name
        """
        img_path, wnid = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # class_name = self.class_names[self.wnids.index(wnid)]
        return image, self.wnids.index(wnid), img_path

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

if __name__ == "__main__":
    VALIDATION_PATH = "/common/datasets/ImageNet_ILSVRC2012/val"
    CLASS_MAPPING_FILE = "/common/datasets/ImageNet_ILSVRC2012/synset_words.txt"
    
    dataset = ImageNetValidationDataset(
        validation_path=VALIDATION_PATH,
        class_mapping_file=CLASS_MAPPING_FILE,
        transform=get_default_transforms()
    )
    image, class_name = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Class name: {class_name}")