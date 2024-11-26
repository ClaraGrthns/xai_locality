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
        transform: PyTorch transforms to apply to images
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
        seed: int = 42
    ):
        """
        Initialize the dataset.
        
        Args:
            validation_path (str): Path to ImageNet validation set
            class_mapping_file (str): Path to synset words mapping file
            num_classes (int): Number of classes to sample (default: 200)
            transform: Optional transforms to apply to images
            seed (int): Random seed for class sampling
        """
        self.transform = transform
        
        # Load class mapping
        self.wnids, self.class_names = self._load_class_mapping(class_mapping_file)
        self.class_mapping = dict(zip(self.wnids, self.class_names))
        
        # Sample classes and create data list
        self.data_list = self._create_data_list(validation_path, num_classes, seed)
    
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
        seed: int
    ) -> List[Tuple[str, str]]:
        """Create list of (image_path, class_name) tuples."""
        random.seed(seed)
        
        class_names = os.listdir(validation_path)
        if num_classes:
            sampled_classes = random.sample(class_names, num_classes)
        else:
            sampled_classes = class_names
        data_list = []
        for class_name in sampled_classes:
            class_path = os.path.join(validation_path, class_name)
            for file_name in os.listdir(class_path):
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
            
        class_name = self.class_names[self.wnids.index(wnid)]
        return image, class_name

# Default transforms for ImageNet
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

# Example usage:
if __name__ == "__main__":
    # Paths
    VALIDATION_PATH = "/common/datasets/ImageNet_ILSVRC2012/val"
    CLASS_MAPPING_FILE = "/common/datasets/ImageNet_ILSVRC2012/synset_words.txt"
    
    # Create dataset
    dataset = ImageNetValidationDataset(
        validation_path=VALIDATION_PATH,
        class_mapping_file=CLASS_MAPPING_FILE,
        transform=get_default_transforms()
    )
    
    # Access a sample
    image, class_name = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Class name: {class_name}")