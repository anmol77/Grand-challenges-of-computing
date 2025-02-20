import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = '/home/agauta01/anmol_work/kvasir-dataset-v2'

# Class names
CLASSES = [
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis',
    'normal-cecum', 'normal-pylorus', 'normal-z-line',
    'polyps', 'ulcerative-colitis'
]

class KvasirDataset(Dataset):
    """Dataset class for Kvasir dataset"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def prepare_data():
    """Prepare image paths and labels, split into train, val, test sets"""
    image_paths = []
    labels = []
    
    # Collect all image paths and labels
    class_counts = {}
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} does not exist!")
            continue
        
        class_images = []
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                class_images.append(img_path)
                
        if len(class_images) > 0:
            image_paths.extend(class_images)
            labels.extend([class_idx] * len(class_images))
            class_counts[class_name] = len(class_images)
        else:
            print(f"Warning: No images found in {class_dir}")
    
    # Print class distribution
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Split data: 80% train, 10% validation, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test):
    """Create data loaders with transformations"""
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = KvasirDataset(X_train, y_train, transform=train_transform)
    val_dataset = KvasirDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = KvasirDataset(X_test, y_test, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def build_model():
    """Build EfficientNetV2 model"""
    # Load pre-trained EfficientNetV2 model
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    
    # Modify the classifier for our classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASSES))
    
    return model.to(DEVICE)

def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def evaluate_model(model, test_loader, y_test):
    """Evaluate model and calculate metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Ensure we have matching lengths
    all_preds = all_preds[:len(y_test)]
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Find unique classes in the test set
    unique_classes = np.unique(all_labels)
    present_classes = [CLASSES[i] for i in unique_classes]
    
    # Detailed classification report
    print("\nClassification Report:")
    try:
        print(classification_report(all_labels, all_preds, target_names=present_classes))
    except ValueError as e:
        print("Error generating classification report with target names.")
        print("Generating report without class names:")
        print(classification_report(all_labels, all_preds))
    
    # Print which classes are present in the test set
    print(f"\nClasses present in test set ({len(unique_classes)}/{len(CLASSES)}):")
    for i, class_name in enumerate(CLASSES):
        status = "✓" if i in unique_classes else "✗"
        print(f"{status} {class_name}")
    
    return accuracy, precision, recall, f1

def main():
    print(f"Using device: {DEVICE}")
    
    # Prepare data
    print("Preparing data...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Build model
    print("Building EfficientNetV2 model...")
    model = build_model()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Print stats
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'New best validation accuracy: {best_val_acc:.2f}%')
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved best model checkpoint')
    
    print("\nTraining completed!")
    
    # Load best model for evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on test set
    evaluate_model(model, test_loader, y_test)

if __name__ == "__main__":
    main()