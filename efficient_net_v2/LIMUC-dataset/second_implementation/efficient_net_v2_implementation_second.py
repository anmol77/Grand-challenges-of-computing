import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

class LIMUCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        print(f"\nInitializing dataset from: {root_dir}")
        if not os.path.exists(root_dir):
            print(f"ERROR: Directory {root_dir} does not exist!")
            return
            
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            print(f"Checking class directory: {class_path}")
            
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith('.bmp')]
                print(f"Found {len(images)} images in {class_name}")
                
                for img_name in images:
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
            else:
                print(f"Warning: {class_path} is not a directory or doesn't exist")

        print(f"Total images loaded for {os.path.basename(root_dir)}: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_metrics(all_labels, all_preds):
    # Calculate per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    # Calculate overall metrics
    overall_precision = np.mean(precision_per_class)
    overall_recall = np.mean(recall_per_class)
    overall_f1 = np.mean(f1_per_class)
    overall_accuracy = (np.array(all_labels) == np.array(all_preds)).mean()
    
    return {
        'per_class': {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class
        },
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'accuracy': overall_accuracy
        }
    }

def main():
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Enhanced data augmentation
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Validation/Test transforms (no augmentation)
        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        base_path = '/home/agauta01/anmol_work/LIMUC-Dataset'
        
        # Create datasets with 80-10-10 split
        train_dataset = LIMUCDataset(os.path.join(base_path, 'train_set'), transform=train_transform)
        val_dataset = LIMUCDataset(os.path.join(base_path, 'validation_set'), transform=eval_transform)
        test_dataset = LIMUCDataset(os.path.join(base_path, 'test_set'), transform=eval_transform)

        if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
            print("Error: One or more datasets are empty!")
            return

        # Calculate class weights for handling imbalance
        labels = np.array(train_dataset.labels)
        class_weights = compute_class_weight('balanced', 
                                            classes=np.unique(labels),
                                            y=labels)
        class_weights = torch.FloatTensor(class_weights).to(device)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        # Initialize model with regularization
        model = torchvision.models.efficientnet_v2_s(pretrained=True)
        num_classes = 4
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        model = model.to(device)

        # Loss function with class weights and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=5, verbose=True
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=7)

        num_epochs = 50
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if i % 10 == 9:
                    print(f'Batch {i+1}, Loss: {running_loss/10:.3f}, Acc: {100*correct/total:.2f}%')
                    running_loss = 0.0

            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            val_preds = []
            val_labels_list = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels_list.extend(labels.cpu().numpy())
            
            val_acc = 100 * val_correct / val_total
            print(f'Validation Accuracy: {val_acc:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Early stopping check
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')

        # Load best model and evaluate
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        
        # Test set evaluation
        test_preds = []
        test_labels_list = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                test_preds.extend(predicted.cpu().numpy())
                test_labels_list.extend(labels.cpu().numpy())
        
        # Calculate and print metrics
        metrics = calculate_metrics(test_labels_list, test_preds)
        
        print("\nTest Results:")
        print("Per-class metrics:")
        for i, class_name in enumerate(['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3']):
            print(f"\n{class_name}:")
            print(f"Precision: {metrics['per_class']['precision'][i]:.4f}")
            print(f"Recall: {metrics['per_class']['recall'][i]:.4f}")
            print(f"F1-score: {metrics['per_class']['f1'][i]:.4f}")
        
        print("\nOverall metrics:")
        print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
        print(f"Precision: {metrics['overall']['precision']:.4f}")
        print(f"Recall: {metrics['overall']['recall']:.4f}")
        print(f"F1-score: {metrics['overall']['f1']:.4f}")

    except Exception as e:
        print(f"Error encountered: {str(e)}")
        raise e

if __name__ == "__main__":
    main()