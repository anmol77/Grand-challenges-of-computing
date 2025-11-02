import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import time
import os
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=8, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, E)
        x = self.proj_drop(self.proj(x))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=8, in_channels=3, num_classes=4,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.2):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        # Added extra layers for classification
        self.pre_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head = nn.Linear(embed_dim // 2, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = x[:, 0]  # Take CLS token
        x = self.pre_head(x)
        x = self.head(x)
        return x

class LIMUCDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith('.bmp'):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def get_class_weights(train_dataset):
    labels = [label for _, label in train_dataset]
    class_weights = compute_class_weight('balanced', 
                                       classes=np.unique(labels), 
                                       y=labels)
    return torch.FloatTensor(class_weights)

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, 
                      scheduler, num_epochs, device, patience=15):
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = 100 * val_correct / val_total
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%')
        
        scheduler.step()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model metrics
            best_metrics = classification_report(all_labels, all_preds, 
                                              target_names=['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3'],
                                              output_dict=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'metrics': best_metrics
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    
    # Load and return the best model metrics
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['metrics'], checkpoint['val_acc']

def main():
    # Modified hyperparameters
    batch_size = 8
    num_epochs = 150
    learning_rate = 1e-4
    weight_decay = 0.05
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])
    
    # Simpler validation transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    

    train_dataset = LIMUCDataset('/home/agauta01/anmol_work/LIMUC-Dataset/train_set', transform=train_transform)
    val_dataset = LIMUCDataset('/home/agauta01/anmol_work/LIMUC-Dataset/validation_set', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Get class weights for balanced loss
    class_weights = get_class_weights(train_dataset).to(device)
    
    # Initialize model with increased capacity
    model = VisionTransformer(
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout=0.2
    ).to(device)
    
    # Add label smoothing to criterion
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=learning_rate, 
                                 weight_decay=weight_decay)
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Train and evaluate
    start_time = time.time()
    best_metrics, best_val_acc = train_and_evaluate(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
    )
    
    # Print final metrics
    print("\nBest Model Performance:")
    print(f"Validation Accuracy: {best_val_acc:.2f}%")
    print("\nDetailed Metrics:")
    for mayo_score in range(4):
        metrics = best_metrics[f'Mayo {mayo_score}']
        print(f"\nMayo {mayo_score}:")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1-score']:.3f}")
    
    print(f"\n3. ------- Calculate time elapsed ------- ")
    print(f"Total of {int(time.time() - start_time)} seconds elapsed for process")

if __name__ == "__main__":
    main()