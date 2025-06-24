import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

import torch
from torch.utils.data import DataLoader
import numpy as np
from dataloading.ptbxl_advanced_dataset import PTBXLAdvancedDataset
from models.ecg_transformer import ECGTransformer
from tools.logger import ExperimentLogger
from tools.checkpoint import save_checkpoint, load_checkpoint
from tools.metrics import multilabel_f1, multilabel_accuracy
import pandas as pd
from ast import literal_eval
from collections import Counter
from multiprocessing import freeze_support
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

CONFIG = {
    'batch_size': 64,
    'lr': 1e-4,
    'epochs': 50,
    'max_len': 5000,
    'input_channels': 12,
    'num_workers': 6,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'd_model': 128,
    'num_layers': 4,
    'num_heads': 8,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'pooling': 'mean',
    'positional_encoding': 'learnable',
    'checkpoint_dir': os.path.join(PROJECT_ROOT, 'checkpoints'),
    'early_stopping_patience': 10,
    'train_csv': os.path.join(PROJECT_ROOT, 'data/processed_ptbxl/train.csv'),
    'val_csv': os.path.join(PROJECT_ROOT, 'data/processed_ptbxl/val.csv'),
    'test_csv': os.path.join(PROJECT_ROOT, 'data/processed_ptbxl/test.csv'),
}

def custom_collate_fn(batch):
    signals, labels, metas = zip(*batch)
    signals = torch.tensor(np.stack(signals), dtype=torch.float32)
    labels = torch.tensor(np.stack(labels), dtype=torch.float32)
    metas = torch.tensor(np.stack(metas), dtype=torch.float32)  # (batch, 8)
    return signals, labels, metas

def main():
    # --- Loading classes ---
    df = pd.read_csv(CONFIG['train_csv'])
    all_labels = []
    for labels in df['labels']:
        all_labels.extend([i for i, v in enumerate(eval(labels)) if v == 1.0])
    # Classes by index, order matches multilabel
    frequent_classes = list(range(len(eval(df['labels'][0]))))
    print('Number of classes:', len(frequent_classes))

    # --- Datasets and DataLoader ---
    train_ds = PTBXLAdvancedDataset(
        csv_path=CONFIG['train_csv'],
        cache_mode=True,
        project_root=PROJECT_ROOT
    )
    val_ds = PTBXLAdvancedDataset(
        csv_path=CONFIG['val_csv'],
        cache_mode=True,
        project_root=PROJECT_ROOT
    )
    test_ds = PTBXLAdvancedDataset(
        csv_path=CONFIG['test_csv'],
        cache_mode=True,
        project_root=PROJECT_ROOT
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], collate_fn=custom_collate_fn)

    # --- Model ---
    model = ECGTransformer(
        input_channels=CONFIG['input_channels'],
        seq_len=CONFIG['max_len'],
        num_classes=len(frequent_classes),
        d_model=CONFIG['d_model'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        dim_feedforward=CONFIG['dim_feedforward'],
        dropout=CONFIG['dropout'],
        pooling=CONFIG['pooling'],
        positional_encoding=CONFIG['positional_encoding'],
        batch_first=True
    )
    device = torch.device(CONFIG['device'])
    model.to(device)

    # --- Optimizer and loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- Logger ---
    logger = ExperimentLogger(log_dir=os.path.join(PROJECT_ROOT, 'logs'), exp_name='ecg_transformer')
    # Log only actual parameters
    log_config = {k: v for k, v in CONFIG.items() if not k.endswith('_csv')}
    logger.log_params(log_config)

    scaler = GradScaler()
    # --- Training ---
    best_val_f1 = 0.0
    best_epoch = 0
    patience = 0
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_losses = []
        for X, y, meta in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", leave=False):
            X, y, meta = X.to(device, non_blocking=True), y.to(device, non_blocking=True), meta.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                logits = model(X, meta)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        logger.log(f'Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f}')

        # --- Validation ---
        model.eval()
        val_losses = []
        all_y = []
        all_logits = []
        with torch.no_grad():
            for X, y, meta in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]", leave=False):
                X, y, meta = X.to(device, non_blocking=True), y.to(device, non_blocking=True), meta.to(device, non_blocking=True)
                with autocast():
                    logits = model(X, meta)
                    loss = criterion(logits, y)
                val_losses.append(loss.item())
                all_y.append(y.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
        all_y = np.concatenate(all_y, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        val_f1 = multilabel_f1(all_y, all_logits)
        val_acc = multilabel_accuracy(all_y, all_logits)
        logger.log_metrics({'val_loss': np.mean(val_losses), 'val_f1': val_f1, 'val_acc': val_acc}, step=epoch+1)
        logger.log(f'Epoch {epoch+1}: val_loss={np.mean(val_losses):.4f}, val_f1={val_f1:.4f}, val_acc={val_acc:.4f}')

        # --- Saving checkpoint ---
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch+1,
            'val_f1': val_f1,
            'config': log_config,
            'classes': frequent_classes,
        }
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
        save_checkpoint(state, CONFIG['checkpoint_dir'], is_best=is_best)

        # --- Early stopping ---
        if patience >= CONFIG['early_stopping_patience']:
            print(f'Early stopping: val_f1 did not improve for {CONFIG["early_stopping_patience"]} epochs.')
            break

    print('Training completed!')

    # --- Testing after training ---
    # Load best checkpoint
    best_ckpt = load_checkpoint(CONFIG['checkpoint_dir'], filename='best.pth', map_location=CONFIG['device'])
    model.load_state_dict(best_ckpt['model_state_dict'])
    optimizer.load_state_dict(best_ckpt['optimizer_state_dict'])
    print(f'Loaded best checkpoint with val_f1 = {best_ckpt["val_f1"]:.4f} (epoch {best_ckpt["epoch"]})')

    model.eval()
    test_losses = []
    all_y = []
    all_logits = []
    with torch.no_grad():
        for X, y, meta in tqdm(test_loader, desc="Test", leave=False):
            X, y, meta = X.to(device, non_blocking=True), y.to(device, non_blocking=True), meta.to(device, non_blocking=True)
            with autocast():
                logits = model(X, meta)
                loss = criterion(logits, y)
            test_losses.append(loss.item())
            all_y.append(y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    all_y = np.concatenate(all_y, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    test_f1 = multilabel_f1(all_y, all_logits)
    test_acc = multilabel_accuracy(all_y, all_logits)
    logger.log_metrics({'test_loss': np.mean(test_losses), 'test_f1': test_f1, 'test_acc': test_acc}, step='test')
    print(f'Test: loss={np.mean(test_losses):.4f}, f1={test_f1:.4f}, acc={test_acc:.4f}')

if __name__ == '__main__':
    freeze_support()
    main() 