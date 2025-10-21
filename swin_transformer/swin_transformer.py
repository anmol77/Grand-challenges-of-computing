import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm  # PyTorch Image Models library
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths - UPDATE THESE FOR YOUR HPC CLUSTER
    data_root = "/home/agauta01/anmol_work/LIMUC-Dataset"  # Root directory containing train/val/test folders
    output_dir = "/home/agauta01/work/swin_transformer_results"
    
    # Model configuration
    model_name = "swin_base_patch4_window12_384"
    pretrained = True
    pretrained_weights_path = "/home/agauta01/work/swin_base_weights/swin_base_patch4_window12_384.pth"  # Local weights file
    num_classes = 4
    img_size = 384  # Swin Base uses 384x384 images
    
    # Training hyperparameters
    batch_size = 16  # Reduced for larger model (was 32 for tiny)
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 0.05
    warmup_epochs = 5
    
    # Learning rate schedule
    min_lr = 1e-6
    
    # Data augmentation
    use_augmentation = True
    
    # System
    num_workers = 4  # Number of CPU workers for data loading
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = True  # Use automatic mixed precision (AMP)
    
    # Early stopping
    patience = 15
    
    # Class names - UPDATE THESE to match your actual folder names
    class_names = ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3']  # Change if your folders are named differently (e.g., 'Mayo0', 'mayo_0', etc.)
    
    # Distributed training (if using multiple GPUs)
    distributed = False  # Set to True for multi-GPU training


# ============================================================================
# DATASET
# ============================================================================
class LIMUCDataset(Dataset):
    """
    Custom dataset for LIMUC ulcerative colitis images.
    Expected directory structure:
    data_root/
        train_set/
            Mayo 0/
            Mayo 1/
            Mayo 2/
            Mayo 3/
        validation_set/
            Mayo 0/
            ...
        test_set/
            Mayo 0/
            ...
    """
    def __init__(self, root_dir, split='train_set', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = Config.class_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"\nLooking for data in: {self.root_dir}")
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            print(f"Checking directory: {class_dir}")
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            # Look for .bmp files (LIMUC dataset format)
            bmp_files = list(class_dir.glob("*.bmp"))
            jpg_files = list(class_dir.glob("*.jpg"))
            png_files = list(class_dir.glob("*.png"))
            
            for img_path in bmp_files:
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
            for img_path in jpg_files:
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
            for img_path in png_files:
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
            
            print(f"  Found {len(bmp_files) + len(jpg_files) + len(png_files)} images in {class_name}")
        
        print(f"\n{split}: {len(self.samples)} total images")
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}. Please check your data path and folder structure.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# DATA AUGMENTATION & PREPROCESSING
# ============================================================================
def get_transforms(split='train'):
    """Get appropriate transforms for train/val/test splits"""
    
    if split == 'train' and Config.use_augmentation:
        return transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


# ============================================================================
# MODEL
# ============================================================================
def create_model():
    """Create Swin Transformer model using timm library"""
    
    print(f"Creating model: {Config.model_name}")
    
    # Load model architecture without pretrained weights first
    model = timm.create_model(
        Config.model_name,
        pretrained=False,  # Don't download from internet
        num_classes=Config.num_classes
    )
    
    # If we have local pretrained weights, load them
    if Config.pretrained and os.path.exists(Config.pretrained_weights_path):
        print(f"Loading pretrained weights from: {Config.pretrained_weights_path}")
        
        # Load the state dict
        state_dict = torch.load(Config.pretrained_weights_path, map_location='cpu')
        
        # Remove ALL classifier head weights (different models use different names)
        # Possible classifier keys: 'head.weight', 'head.bias', 'head.fc.weight', 'head.fc.bias'
        keys_to_remove = [k for k in state_dict.keys() if k.startswith('head.')]
        
        for key in keys_to_remove:
            print(f"  Removing classifier weight: {key}")
            del state_dict[key]
        
        # Load weights (strict=False allows loading even with missing classifier)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded pretrained backbone weights!")
        print(f"Missing keys (classifier head - expected): {[k for k in msg.missing_keys if 'head' in k]}")
    elif Config.pretrained:
        print(f"Warning: Pretrained weights not found at {Config.pretrained_weights_path}")
        print("Training from scratch instead...")
    else:
        print("Training from scratch (no pretrained weights)")
    
    return model


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if Config.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================
def cosine_scheduler(optimizer, epoch, max_epochs, base_lr, min_lr, warmup_epochs):
    """Cosine learning rate schedule with warmup"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1 + np.cos(np.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs))
        )
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def main():
    # Create output directory
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Save configuration
    config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('_')}
    with open(os.path.join(Config.output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4, default=str)
    
    # Set device
    device = torch.device(Config.device)
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = LIMUCDataset(Config.data_root, split='train_set', 
                                 transform=get_transforms('train'))
    val_dataset = LIMUCDataset(Config.data_root, split='validation_set', 
                               transform=get_transforms('val'))
    test_dataset = LIMUCDataset(Config.data_root, split='test_set', 
                                transform=get_transforms('test'))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, 
                             shuffle=True, num_workers=Config.num_workers, 
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, 
                           shuffle=False, num_workers=Config.num_workers, 
                           pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, 
                            shuffle=False, num_workers=Config.num_workers, 
                            pin_memory=True)
    
    # Calculate class weights for handling imbalance
    print("\nCalculating class weights for imbalanced dataset...")
    train_labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(train_labels)
    
    print("Class distribution in training set:")
    for i, (class_name, count) in enumerate(zip(Config.class_names, class_counts)):
        print(f"  {class_name}: {count} images ({count/len(train_labels)*100:.1f}%)")
    
    # Compute balanced class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print("\nClass weights (higher weight = more emphasis on minority class):")
    for class_name, weight in zip(Config.class_names, class_weights):
        print(f"  {class_name}: {weight:.3f}")
    
    # Create model
    model = create_model()
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, 
                           weight_decay=Config.weight_decay)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if Config.mixed_precision else None
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{Config.num_epochs}")
        
        # Adjust learning rate
        current_lr = cosine_scheduler(
            optimizer, epoch, Config.num_epochs, 
            Config.learning_rate, Config.min_lr, Config.warmup_epochs
        )
        print(f"Learning rate: {current_lr:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, scaler, device)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, os.path.join(Config.output_dir, 'best_model.pth'))
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Save training history
    np.save(os.path.join(Config.output_dir, 'history.npy'), history)
    
    # Test on best model
    print("\n" + "=" * 80)
    print("Testing best model...")
    checkpoint = torch.load(os.path.join(Config.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, 
                                                            criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                               target_names=Config.class_names))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=Config.class_names, 
                yticklabels=Config.class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.output_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    print(f"\nTraining complete! Results saved to {Config.output_dir}")


if __name__ == '__main__':
    main()