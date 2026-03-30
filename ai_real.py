import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

# ==========================================
# 1. Hardware Routing
# ==========================================
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"Using device: {device}")

# ==========================================
# 2. Data Pipeline & K-Fold Setup
# ==========================================
# Transforms: Resize, Convert to Tensor, and Normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the full dataset (master_dataset -> Ai_generated_dataset / real_dataset -> categories)
data_dir = './master_dataset'
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
print(f"Classes mapping: {full_dataset.class_to_idx}")
print(f"Total images loaded: {len(full_dataset)}")

# Configure 5-Fold Cross-Validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = {}

# ==========================================
# 3. Model Architecture (3 Hidden Layers)
# ==========================================
class AICheckerCNN(nn.Module):
    def __init__(self):
        super(AICheckerCNN, self).__init__()
        
        # Hidden Layer 1: Input 3 channels -> 16 channels, Size: 224 -> 112
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Hidden Layer 2: Input 16 channels -> 32 channels, Size: 112 -> 56
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Hidden Layer 3: Input 32 channels -> 64 channels, Size: 56 -> 28
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Output Layer: Flattened size is 64 channels * 28 * 28 spatial dimensions
        self.fc = nn.Linear(in_features=64 * 28 * 28, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Flatten tensor while keeping the batch dimension intact
        x = x.view(x.size(0), -1) 
        
        # Raw logits output
        x = self.fc(x)
        return x

# ==========================================
# 4. Cross-Validation & Training Loop
# ==========================================
epochs = 20

for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    print(f'\n{"="*10} FOLD {fold + 1}/{k_folds} {"="*10}')
    
    # Create DataLoaders for the current fold
    train_sub = Subset(full_dataset, train_idx)
    val_sub = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)
    
    # Initialize a fresh model, optimizer, and loss function for each fold
    model = AICheckerCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training Loop for current fold
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            # Crucial: Reshape labels to [batch_size, 1] and convert to float
            labels = labels.view(-1, 1).float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Training Loss: {running_loss/len(train_loader):.4f}")
        
    # ==========================================
    # 5. Evaluation Loop (Metrics: Acc, MAE, RMSE)
    # ==========================================
    model.eval()
    fold_preds = []
    fold_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.view(-1, 1).float().to(device)
            
            outputs = model(images)
            # Apply sigmoid to get probabilities for MAE/RMSE calculation
            probs = torch.sigmoid(outputs)
            
            fold_preds.extend(probs.cpu().numpy())
            fold_labels.extend(labels.cpu().numpy())
            
    # Calculate fold metrics
    fold_preds = np.array(fold_preds)
    fold_labels = np.array(fold_labels)
    
    # Mean Absolute Error and Root Mean Squared Error on probabilities
    mae = np.mean(np.abs(fold_preds - fold_labels))
    rmse = np.sqrt(np.mean((fold_preds - fold_labels)**2))
    
    # Accuracy on hard predictions (threshold = 0.5)
    binary_preds = (fold_preds >= 0.5).astype(float)
    accuracy = np.mean(binary_preds == fold_labels) * 100
    
    fold_results[fold] = {"Accuracy": accuracy, "MAE": mae, "RMSE": rmse}
    print(f"Fold {fold+1} Validation -> Accuracy: {accuracy:.2f}%, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# ==========================================
# 6. Final Summary
# ==========================================
print("\n" + "="*40)
print("FINAL CROSS-VALIDATION SUMMARY (5 Folds)")
print("="*40)

avg_acc = np.mean([fold_results[f]["Accuracy"] for f in fold_results])
avg_mae = np.mean([fold_results[f]["MAE"] for f in fold_results])
avg_rmse = np.mean([fold_results[f]["RMSE"] for f in fold_results])

print(f"Average Accuracy : {avg_acc:.2f}%")
print(f"Average MAE      : {avg_mae:.4f}")
print(f"Average RMSE     : {avg_rmse:.4f}")
print("="*40)