import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pathlib
class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Define subdirectories
        self.images_dir = os.path.join(root_dir, 'leftImg8bit_trainvaltest', split)
        self.labels_dir = os.path.join(root_dir, 'gtFine', split)
        
        # List of image files
        self.images = []
        for city in os.listdir(self.images_dir):
            city_path = os.path.join(self.images_dir, city)
            if os.path.isdir(city_path):
                for filename in os.listdir(city_path):
                    if filename.endswith('_leftImg8bit.png'):
                        self.images.append(os.path.join(city, filename))
        
        print(f"Found {len(self.images)} images in {split} set")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Get corresponding label path
        label_name = img_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        label_path = os.path.join(self.labels_dir, label_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            label = Image.open(label_path)
            
            if self.transform:
                image = self.transform(image)
                label = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(label)
                label = transforms.ToTensor()(label).squeeze(0).long()
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path} or label {label_path}: {e}")
            raise e

def get_data_loaders(root_dir, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CityscapesDataset(root_dir=root_dir, split='train', transform=transform)
    val_dataset = CityscapesDataset(root_dir=root_dir, split='val', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,pin_memory=False)

    
    return train_loader, val_loader

# Test the data loader
if __name__ == "__main__":
    root_dir = "D:/Research Internship IISER Bhopal/proto-segmentation-master sacha/proto-segmentation-master/CityScapes Dataset"
    train_loader, val_loader = get_data_loaders(root_dir)
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        break
