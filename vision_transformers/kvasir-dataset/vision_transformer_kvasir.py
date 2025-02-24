import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import pandas as pd
from datetime import datetime

# Setting seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 65
NUM_CLASSES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4

# Path to dataset
dataset_path = "/home/agauta01/anmol_work/kvasir-dataset-v2"

# Class names
class_names = [
    "dyed-lifted-polyps", "dyed-resection-margins", "esophagitis", 
    "normal-cecum", "normal-pylorus", "normal-z-line", "polyps", "ulcerative-colitis"
]

class KvasirDataset(Dataset):
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

def get_data():
    # Create empty lists to store images and labels
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                images.append(img_path)
                labels.append(class_idx)
    
    return np.array(images), np.array(labels)

def build_datasets():
    # Get data
    print("Loading data...")
    images, labels = get_data()
    
    # Split data into training, validation and test sets (80-10-10)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.111, random_state=42, stratify=y_train_val
    )
    
    print(f"Training set: {len(x_train)} samples")
    print(f"Validation set: {len(x_val)} samples")
    print(f"Test set: {len(x_test)} samples")
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = KvasirDataset(x_train, y_train, transform=train_transform)
    val_dataset = KvasirDataset(x_val, y_val, transform=val_test_transform)
    test_dataset = KvasirDataset(x_test, y_test, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, x_test, y_test

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        # (batch_size, channels, height, width) -> (batch_size, embed_dim, grid, grid)
        x = self.proj(x)
        # (batch_size, embed_dim, grid, grid) -> (batch_size, embed_dim, n_patches)
        x = x.flatten(2)
        # (batch_size, embed_dim, n_patches) -> (batch_size, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        # x: (batch_size, n_patches + 1, dim)
        batch_size, n_tokens, dim = x.shape
        
        qkv = self.qkv(x).reshape(
            batch_size, n_tokens, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        # qkv: (3, batch_size, n_heads, n_patches + 1, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v: (batch_size, n_heads, n_patches + 1, head_dim)
        
        # (batch_size, n_heads, n_patches + 1, n_patches + 1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # (batch_size, n_heads, n_patches + 1, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, n_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
            drop=drop
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        n_classes=NUM_CLASSES,
        embed_dim=768, 
        depth=12,
        n_heads=12, 
        mlp_ratio=4.0, 
        qkv_bias=True,
        drop_rate=0.1, 
        attn_drop_rate=0.0
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        # Normalization and classification head
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize patch embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize all other weights
        self.apply(self._init_weights_general)
    
    def _init_weights_general(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        # (batch_size, channels, img_size, img_size) -> (batch_size, n_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply normalization
        x = self.norm(x)
        
        # Return class token features
        return x[:, 0]
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = len(train_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS} (Train) - Starting...")
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress every 10 batches or at the end
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == batch_count:
            current_loss = running_loss / total
            current_acc = 100. * correct / total
            print(f"  Batch {batch_idx+1}/{batch_count}: loss={current_loss:.4f}, acc={current_acc:.2f}%")
    
    train_loss = running_loss / total
    train_accuracy = 100. * correct / total
    
    print(f"Epoch {epoch+1}/{EPOCHS} (Train) - Completed: loss={train_loss:.4f}, acc={train_accuracy:.2f}%")
    return train_loss, train_accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print("Validation - Starting...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / total
    val_accuracy = 100. * correct / total
    
    print(f"Validation - Completed: val_loss={val_loss:.4f}, val_acc={val_accuracy:.2f}%")
    return val_loss, val_accuracy



def evaluate_model(model, test_loader, device, x_test, y_test):
    model.eval()
    all_preds = []
    all_targets = []
    
    print("Evaluating model on test set...")
    batch_count = len(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processing test batch {batch_idx+1}/{batch_count}")
                
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    report = classification_report(
        all_targets, all_preds,
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    
    # Print classification report
    print("\nClassification Report:")
    report_df = pd.DataFrame(report).transpose()
    print(report_df.to_string())
    
    # Calculate overall metrics
    accuracy = report['accuracy']
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']
    
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    
    # Save confusion matrix as CSV
    cm = confusion_matrix(all_targets, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv('vit_confusion_matrix.csv')
    print("Confusion matrix saved to 'vit_confusion_matrix.csv'")
    
    return accuracy, macro_precision, macro_recall, macro_f1, report_df

def train_model():
    # Build datasets and dataloaders
    train_loader, val_loader, test_loader, x_test, y_test = build_datasets()
    
    # Create model with custom configuration
    print(f"Creating Vision Transformer model on {DEVICE}...")
    model = VisionTransformer(
        img_size=IMAGE_SIZE,
        patch_size=16,
        in_channels=3,
        n_classes=NUM_CLASSES,
        embed_dim=768,
        depth=12,
        n_heads=12,
        drop_rate=0.1
    )
    model = model.to(DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_path = 'kvasir_vit_best_model.pth'
    
    # Training loop
    print(f"Starting training on {DEVICE}...")
    start_time = datetime.now()
    
    for epoch in range(EPOCHS):
        # Train one epoch
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, DEVICE
        )
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, DEVICE)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        history['learning_rate'].append(current_lr)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
              f"LR: {current_lr:.6f}")
        
        # Check if we should save the model (best validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = datetime.now() - start_time
    print(f"Training completed in {training_time}")
    
    # Save final training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('vit_final_training_history.csv', index=False)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate on test set
    print("\nEvaluating best model on test set...")
    accuracy, precision, recall, f1, report_df = evaluate_model(
        model, test_loader, DEVICE, x_test, y_test
    )
    
    # Save detailed results to CSV
    report_df.to_csv("vit_classification_report.csv")
    
    # Save predictions on test set
    test_predictions = []
    test_ground_truth = []
    test_probabilities = []
    
    model.eval()
    batch_count = len(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processing prediction batch {batch_idx+1}/{batch_count}")
                
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            test_predictions.extend(preds.cpu().numpy())
            test_ground_truth.extend(targets.cpu().numpy())
            test_probabilities.extend(probs.cpu().numpy())
    
    # No need to save test predictions to CSV
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': class_names,
        'history': history,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'training_time': str(training_time),
        'epochs_completed': epoch + 1
    }, 'kvasir_vit_final_model.pth')
    
    print("Model and results saved successfully.")
    return model, history

if __name__ == "__main__":
    model, history = train_model()