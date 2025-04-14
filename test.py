import os
import torch
import timm
import torch.nn as nn
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from dataset import DiabeticRetinopathyDataset  # assuming you refactored the dataset class to a separate file
import logging
import os
from args import get_test_args
args = get_test_args()

# Create log directory
log_dir = "test_logs"
os.makedirs(log_dir, exist_ok=True)

# Log file based on model name (no timestamp)
log_filename = f"test_{args.model_name}.log"
log_path = os.path.join(log_dir, log_filename)

# Configure logging: to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler()
    ]
)

logging.info(f"🔍 Starting evaluation for model: {args.model_name}")
logging.info(f"📂 Logs will be saved to: {log_path}")


# Load model
model = timm.create_model(args.model_name, pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 5)
model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
model.eval()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loss
try:
    loss_class = getattr(nn, args.loss)
    criterion = loss_class()
except AttributeError:
    raise ValueError(f"Loss function '{args.loss}' not found in torch.nn")

# Dataset & DataLoader
test_dataset = DiabeticRetinopathyDataset(img_dir=args.img_dir, split='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Evaluation
test_loss = 0.0
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"\n🔍 Test Loss: {test_loss/len(test_loader):.4f}")
print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")
logging.info(f"\n🔍 Test Loss: {test_loss/len(test_loader):.4f}")
logging.info(f"✅ Test Accuracy: {100 * correct / total:.2f}%")

