import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from config import COMBINE_PREPROCESSED_CNN_PATH
import copy

class CNN1D(nn.Module):
    """1D CNN for tabular data classification"""
    def __init__(self, input_dim, num_classes=5):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Calculate the flattened size
        self.flatten_size = 256 * (input_dim // 8)

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Add channel dimension: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)

        # Convolutional blocks
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

def load_and_prepare_data():
    """Load data and prepare features and labels"""
    df = pd.read_csv(COMBINE_PREPROCESSED_CNN_PATH)

    # Remove Sample File column
    if 'Sample File' in df.columns:
        df = df.drop('Sample File', axis=1)

    # Separate features and labels
    X = df.drop('target_noc', axis=1)
    y = df['target_noc']

    # Convert labels to 0-indexed (1-5 -> 0-4)
    y = y - 1

    return X, y

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    """Evaluate the model"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    return np.array(all_labels), np.array(all_preds)

def evaluate_model():
    """Train CNN model with 5-fold cross-validation and evaluate"""
    X, y = load_and_prepare_data()

    # Initialize 5-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Store metrics for each fold
    metrics = {
        'micro_precision': [],
        'macro_precision': [],
        'micro_recall': [],
        'macro_recall': [],
        'micro_f1': [],
        'macro_f1': []
    }

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Starting 5-fold cross-validation...")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/5")
        print("-" * 60)

        # Split data into train and test
        X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Further split train into train (90%) and validation (10%) for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
        )

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train.values)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val.values)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test.values)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model
        input_dim = X_train.shape[1]
        model = CNN1D(input_dim=input_dim, num_classes=5).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop with early stopping
        num_epochs = 200
        patience = 20
        best_f1 = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            # Evaluate on validation set
            y_val_true, y_val_pred = evaluate(model, val_loader, device)
            val_f1 = f1_score(y_val_true, y_val_pred, average='macro')

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val Macro F1: {val_f1:.4f}")

            # Early stopping check
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}, Best Val Macro F1: {best_f1:.4f}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Evaluate on test set
        y_true, y_pred = evaluate(model, test_loader, device)

        # Calculate metrics
        micro_prec = precision_score(y_true, y_pred, average='micro')
        macro_prec = precision_score(y_true, y_pred, average='macro')
        micro_rec = recall_score(y_true, y_pred, average='micro')
        macro_rec = recall_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        # Store metrics
        metrics['micro_precision'].append(micro_prec)
        metrics['macro_precision'].append(macro_prec)
        metrics['micro_recall'].append(micro_rec)
        metrics['macro_recall'].append(macro_rec)
        metrics['micro_f1'].append(micro_f1)
        metrics['macro_f1'].append(macro_f1)

        # Print fold results
        print(f"Micro Precision: {micro_prec:.4f}")
        print(f"Macro Precision: {macro_prec:.4f}")
        print(f"Micro Recall:    {micro_rec:.4f}")
        print(f"Macro Recall:    {macro_rec:.4f}")
        print(f"Micro F1:        {micro_f1:.4f}")
        print(f"Macro F1:        {macro_f1:.4f}")

    # Calculate and print average metrics
    print("\n" + "=" * 60)
    print("AVERAGE PERFORMANCE ACROSS 5 FOLDS")
    print("=" * 60)
    print(f"Micro Precision: {np.mean(metrics['micro_precision']):.4f} ± {np.std(metrics['micro_precision']):.4f}")
    print(f"Macro Precision: {np.mean(metrics['macro_precision']):.4f} ± {np.std(metrics['macro_precision']):.4f}")
    print(f"Micro Recall:    {np.mean(metrics['micro_recall']):.4f} ± {np.std(metrics['micro_recall']):.4f}")
    print(f"Macro Recall:    {np.mean(metrics['macro_recall']):.4f} ± {np.std(metrics['macro_recall']):.4f}")
    print(f"Micro F1:        {np.mean(metrics['micro_f1']):.4f} ± {np.std(metrics['micro_f1']):.4f}")
    print(f"Macro F1:        {np.mean(metrics['macro_f1']):.4f} ± {np.std(metrics['macro_f1']):.4f}")
    print("=" * 60)

    return metrics

if __name__ == "__main__":
    metrics = evaluate_model()
