from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, img_dir, split='train', transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            split (string): The split to use ('train', 'val', or 'test').
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir 
        self.transform = transform  

        self.image_paths = []
        self.labels = []
        self.classes = ['No_DR', 'Mild', 'Moderate', 'Proliferate_DR', 'Severe']

        for i, class_name in enumerate(self.classes):
            class_folder = os.path.join(img_dir, class_name)
            for img_name in os.listdir(class_folder):
                self.image_paths.append(os.path.join(class_folder, img_name))
                self.labels.append(i)  # Assign label based on folder name

        # Split the data into train, validation, and test sets
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(self.image_paths, self.labels, test_size=0.2, random_state=42)
        train_paths, val_paths, train_labels, val_labels = train_test_split(train_val_paths, train_val_labels, test_size=0.1, random_state=42)

        # Assign the appropriate split based on the argument
        if split == 'train':
            self.image_paths = train_paths
            self.labels = train_labels
        elif split == 'val':
            self.image_paths = val_paths
            self.labels = val_labels
        elif split == 'test':
            self.image_paths = test_paths
            self.labels = test_labels
        else:
            raise ValueError("split must be one of 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.image_paths)  # Number of samples

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]  # Get the image path
        image = Image.open(img_path).convert('RGB')  # Open the image in RGB mode

        label = self.labels[idx]  # Get the label

        if self.transform:
            image = self.transform(image)  # Apply transformations to the image

        return image, label  # Return image and label as a tuple