import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from tabpfn import TabPFNClassifier
from config import COMBINE_PREPROCESSED_PATH
import torch
def load_and_prepare_data():
    """Load data and prepare features and labels"""
    df = pd.read_csv(COMBINE_PREPROCESSED_PATH)

    # Remove Sample File column
    if 'Sample File' in df.columns:
        df = df.drop('Sample File', axis=1)

    # Separate features and labels
    X = df.drop('target_noc', axis=1)
    y = df['target_noc']

    # Convert labels to 0-indexed for TabPFN (1-5 -> 0-4)
    y = y - 1

    return X, y

def evaluate_model():
    """Train TabPFN model with 5-fold cross-validation and evaluate"""
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

    print("Starting 5-fold cross-validation...")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/5")
        print("-" * 60)

        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Initialize and train TabPFN classifier
        model = TabPFNClassifier(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            ignore_pretraining_limits=True
        )

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_val)

        # Calculate metrics
        micro_prec = precision_score(y_val, y_pred, average='micro')
        macro_prec = precision_score(y_val, y_pred, average='macro')
        micro_rec = recall_score(y_val, y_pred, average='micro')
        macro_rec = recall_score(y_val, y_pred, average='macro')
        micro_f1 = f1_score(y_val, y_pred, average='micro')
        macro_f1 = f1_score(y_val, y_pred, average='macro')

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
