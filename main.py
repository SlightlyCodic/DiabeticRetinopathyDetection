import os
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from dataset import DiabeticRetinopathyDataset
from args import get_train_args
import logging
from datetime import datetime
args = get_train_args()

# Create logs directory
# Build log file name with model name (no timestamp)
log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)

log_filename = f"train_{args.model_name}.log"
log_path = os.path.join(log_dir, log_filename)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='w'),  # Save to file
        logging.StreamHandler()  # Print to console
    ]
)

logging.info(f"🔧 Starting training for model: {args.model_name}")
logging.info(f"Saving log to: {log_path}")


# Data augmentation and preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to a fixed size
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(30),  # Random rotation
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

# Load the EfficientNet-B7 model and modify it for 5 classes
model = timm.create_model(args.model_name, pretrained=True)  # EfficientNet B7 pre-trained model
num_ftrs = model.classifier.in_features  
model.classifier = nn.Linear(num_ftrs, 5) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer
try:
    optimizer_class = getattr(optim, args.optimizer)
    optimizer = optimizer_class(model.parameters(), lr=args.lr)
except AttributeError:
    raise ValueError(f"Optimizer '{args.optimizer}' is not found in torch.optim")

# Loss Function
try:
    loss_class = getattr(nn, args.loss)
    criterion = loss_class()
except AttributeError:
    raise ValueError(f"Loss function '{args.loss}' is not found in torch.nn")


# Create datasets for train, validation, and test
img_dir = args.img_dir  # Adjust this path
train_dataset = DiabeticRetinopathyDataset(img_dir=args.img_dir, split='train',transform=transform)
val_dataset = DiabeticRetinopathyDataset(img_dir=args.img_dir, split='val',transform=transform)
test_dataset = DiabeticRetinopathyDataset(img_dir=args.img_dir, split='test',transform=transform)

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Training loop
epochs = args.epochs
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagate the gradients

        # Update the weights
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print statistics for this epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients for validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    logging.info(f"\n📅 Epoch {epoch+1}/{epochs}")
    logging.info(f"📈 Train Loss: {running_loss/len(train_loader):.4f} | Train Accuracy: {100 * correct / total:.2f}%")
    logging.info(f"🧪 Validation Loss: {val_loss/len(val_loader):.4f} | Validation Accuracy: {100 * correct / total:.2f}%")


# Save the trained model
model_dir = os.path.dirname(args.model_path)
if model_dir and not os.path.exists(model_dir):
    os.makedirs(model_dir)
torch.save(model.state_dict(), args.model_path)
print("Model saved successfully!")
