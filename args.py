import argparse

def get_train_args():
    parser = argparse.ArgumentParser(description='Train EfficientNet for Diabetic Retinopathy Detection')

    parser.add_argument('--model_name', type=str, required=True, help='Model architecture from timm (e.g., efficientnet_b0)')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--model_path', type=str, default='models/EfficientNetB0.pth', help='Path to save model')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer class name from torch.optim')
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='Loss class name from torch.nn')

    return parser.parse_args()

def get_test_args():
    parser = argparse.ArgumentParser(description='Evaluate EfficientNet on Diabetic Retinopathy Test Set')

    parser.add_argument('--model_name', type=str, required=True, help='Model architecture from timm (e.g., efficientnet_b0)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='Loss function from torch.nn')

    return parser.parse_args()
