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
import multiprocessing
from sklearn.metrics import f1_score, classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json

# --- FocalLoss реализация ---
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# --- Asymmetric Loss ---
class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction='mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
    def forward(self, logits, targets):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        xs_pos = torch.clamp(xs_pos, min=self.eps, max=1-self.eps)
        xs_neg = torch.clamp(xs_neg, min=self.eps, max=1-self.eps)
        loss_pos = targets * torch.log(xs_pos)
        loss_neg = (1 - targets) * torch.log(xs_neg)
        if self.clip is not None and self.clip > 0:
            loss_neg = torch.clamp(loss_neg, min=-self.clip)
        loss = loss_pos + loss_neg
        pt0 = xs_pos * targets
        pt1 = xs_neg * (1 - targets)
        pt = pt0 + pt1
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        loss *= (1 - pt) ** gamma
        loss = -loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

# --- Threshold optimization ---
def optimize_thresholds(y_true, probs):
    thresholds = np.zeros(probs.shape[1])
    for i in range(probs.shape[1]):
        best_thr = 0.5
        best_f1 = 0
        for thr in np.linspace(0.05, 0.95, 19):
            f1 = f1_score(y_true[:, i], (probs[:, i] > thr).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        thresholds[i] = best_thr
    return thresholds

# --- Новые папки для эксперимента v2 ---
EXPERIMENT_NAME = 'v2'
LOGS_DIR = os.path.join(PROJECT_ROOT, f'logs_{EXPERIMENT_NAME}')
CKPT_DIR = os.path.join(PROJECT_ROOT, f'checkpoints_{EXPERIMENT_NAME}')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, f'artifacts_{EXPERIMENT_NAME}')
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

CONFIG = {
    'batch_size': 32,
    'lr': 1e-5,
    'epochs': 30,
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
    'checkpoint_dir': CKPT_DIR,
    'early_stopping_patience': 10,
    'train_csv': os.path.join(PROJECT_ROOT, 'data/processed_ptbxl/train_filtered.csv'),
    'val_csv': os.path.join(PROJECT_ROOT, 'data/processed_ptbxl/val_filtered.csv'),
    'test_csv': os.path.join(PROJECT_ROOT, 'data/processed_ptbxl/test_filtered.csv'),
    'resume': True,
    'loss_type': 'bce',  # 'bce', 'focal', 'asymmetric'
}

def custom_collate_fn(batch):
    # Фильтруем None (например, если файл не найден)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # или можно бросить ошибку
    signals, labels, metas = zip(*batch)
    signals = torch.tensor(np.stack(signals), dtype=torch.float32)
    labels = torch.tensor(np.stack(labels), dtype=torch.float32)
    metas = torch.tensor(np.stack(metas), dtype=torch.float32)  # (batch, 8)
    return signals, labels, metas

def auto_tune_batch_size(model, train_ds, device, base_bs=64, max_bs=1024):
    bs = base_bs
    while bs <= max_bs:
        try:
            loader = DataLoader(
                train_ds, batch_size=bs, shuffle=True, num_workers=0, collate_fn=custom_collate_fn, pin_memory=True
            )
            batch = next(iter(loader))
            if batch is None:
                bs //= 2
                continue
            X, y, meta = batch
            X, y, meta = X.to(device), y.to(device), meta.to(device)
            with torch.no_grad():
                _ = model(X, meta)
            del loader, batch, X, y, meta
            torch.cuda.empty_cache()
            print(f'[auto_tune] Подходит batch_size={bs}')
            return bs
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f'[auto_tune] OOM при batch_size={bs}, уменьшаю вдвое...')
                bs //= 2
                torch.cuda.empty_cache()
            else:
                raise e
    print('[auto_tune] Не удалось подобрать batch_size, использую 16')
    return 16

def main():
    # --- Logger ---
    logger = ExperimentLogger(log_dir=LOGS_DIR, exp_name=EXPERIMENT_NAME, to_console=True)
    # --- Loading classes ---
    df = pd.read_csv(CONFIG['train_csv'])
    all_labels = []
    for labels in df['labels']:
        all_labels.extend([i for i, v in enumerate(eval(labels)) if v == 1.0])
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
    print('Train size:', len(train_ds))
    print('Val size:', len(val_ds))
    print('Test size:', len(test_ds))
    if len(train_ds) > 0:
        row = train_ds.df.iloc[0]
        sig_path = row['signal_path']
        if not os.path.isabs(sig_path):
            sig_path = os.path.join(train_ds.project_root, sig_path)
        print('Пример пути к сигналу:', sig_path)
        print('Файл существует?', os.path.exists(sig_path))

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

    # --- Автоматический подбор batch_size и num_workers ---
    if device.type == 'cuda':
        auto_bs = auto_tune_batch_size(model, train_ds, device, base_bs=CONFIG['batch_size'])
        CONFIG['batch_size'] = auto_bs
    CONFIG['num_workers'] = min(8, multiprocessing.cpu_count())
    print(f'[auto_tune] Итоговые параметры: batch_size={CONFIG["batch_size"]}, num_workers={CONFIG["num_workers"]}')

    # --- Логгирование распределения классов ---
    train_labels = np.stack(train_ds.df['labels'].apply(eval).values)
    val_labels = np.stack(val_ds.df['labels'].apply(eval).values)
    logger.log(f'Train class distribution: {np.sum(train_labels, axis=0).tolist()}')
    logger.log(f'Val class distribution: {np.sum(val_labels, axis=0).tolist()}')
    # --- Class weights ---
    class_counts = np.sum(train_labels, axis=0)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=CONFIG['device'])

    # --- Optimizer and loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    if CONFIG.get('loss_type', 'bce') == 'focal':
        criterion = FocalLoss()
    elif CONFIG.get('loss_type', 'bce') == 'asymmetric':
        criterion = AsymmetricLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # --- Resume from checkpoint (дообучение) ---
    start_epoch = 0
    total_epochs = CONFIG['epochs']
    ckpt_path = os.path.join(CONFIG['checkpoint_dir'], 'best.pth')
    if CONFIG.get('resume', False) and os.path.exists(ckpt_path):
        best_ckpt = load_checkpoint(CONFIG['checkpoint_dir'], filename='best.pth', map_location=CONFIG['device'])
        model.load_state_dict(best_ckpt['model_state_dict'])
        optimizer.load_state_dict(best_ckpt['optimizer_state_dict'])
        start_epoch = best_ckpt.get('epoch', 0)
        print(f'Продолжаем обучение с эпохи {start_epoch}, val_f1={best_ckpt.get("val_f1", 0):.4f}')
        print(f'Будет дообучено ещё {total_epochs} эпох (с {start_epoch+1} по {start_epoch+total_epochs})')
    else:
        print(f'Обучение с нуля: {total_epochs} эпох (с 1 по {total_epochs})')

    # --- DataLoader с ускорением ---
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'],
        collate_fn=custom_collate_fn, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'],
        collate_fn=custom_collate_fn, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'],
        collate_fn=custom_collate_fn, pin_memory=True, persistent_workers=True
    )

    scaler = GradScaler()
    # --- Sanity test перед стартом обучения ---
    logits = torch.randn(8, 28) * 0.2
    probs = torch.sigmoid(logits)
    preds = (probs > 0.3).float()
    print("[TEST] Nonzero preds per sample:", preds.sum(dim=1))
    if (preds.sum(dim=1) == 0).all():
        print("[ERROR] Все preds нули даже на случайных логитах с threshold=0.3! Пересмотри threshold или pipeline.")
        exit(1)

    # --- Training ---
    best_val_f1 = 0.0
    best_epoch = 0
    patience = 0
    for epoch in range(start_epoch, start_epoch + total_epochs):
        model.train()
        train_losses = []
        all_y, all_logits, all_meta = [], [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", leave=False):
            if batch is None:
                continue
            X, y, meta = batch
            X, y, meta = X.to(device, non_blocking=True), y.to(device, non_blocking=True), meta.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                logits = model(X, meta)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            all_y.append(y.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
            all_meta.append(meta.detach().cpu().numpy())
        if len(train_losses) == 0:
            logger.log(f'[WARNING] Нет обучающих данных для эпохи {epoch+1} (все файлы отсутствуют или пропущены)')
            continue
        logger.log(f'Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f}')

        # --- TRAIN classification_report ---
        train_y = np.concatenate(all_y, axis=0)
        train_logits = np.concatenate(all_logits, axis=0)
        train_probs = torch.sigmoid(torch.tensor(train_logits)).numpy()
        train_preds_03 = (train_probs > 0.3).astype(int)
        macro_f1_train = f1_score(train_y, train_preds_03, average='macro', zero_division=0)
        per_class_f1_train = f1_score(train_y, train_preds_03, average=None, zero_division=0)
        logger.log(f'TRAIN: macro_f1@0.3={macro_f1_train:.4f}, per_class_f1={per_class_f1_train}')
        logger.log(f'TRAIN: classification_report@0.3:\n{classification_report(train_y, train_preds_03, zero_division=0)}')
        nonzero_pred_rate_train = (train_preds_03.sum(axis=1) > 0).mean()
        logger.log(f"[DEBUG] TRAIN: Процент сэмплов с хотя бы 1 предсказанным классом: {nonzero_pred_rate_train:.2%}")
        per_class_pred_rate_train = train_preds_03.mean(axis=0)
        logger.log(f"[DEBUG] TRAIN: Per-class pred rate: {per_class_pred_rate_train}")
        np.save(os.path.join(ARTIFACTS_DIR, f'train_y_epoch{epoch+1}.npy'), train_y)
        np.save(os.path.join(ARTIFACTS_DIR, f'train_logits_epoch{epoch+1}.npy'), train_logits)
        # Confusion matrix (топ-5 классов)
        mcm = multilabel_confusion_matrix(train_y, train_preds_03)
        top5 = np.argsort(np.sum(train_y, axis=0))[::-1][:5]
        for i in top5:
            fig, ax = plt.subplots()
            cmat = mcm[i]
            im = ax.imshow(cmat, cmap='Blues')
            ax.set_title(f'Train Confusion Matrix class {i}')
            plt.colorbar(im, ax=ax)
            fig.savefig(os.path.join(ARTIFACTS_DIR, f'train_confmat_epoch{epoch+1}_class{i}.png'))
            plt.close(fig)

        # --- Validation ---
        model.eval()
        val_losses = []
        all_y = []
        all_logits = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]", leave=False):
                if batch is None:
                    continue
                X, y, meta = batch
                X, y, meta = X.to(device, non_blocking=True), y.to(device, non_blocking=True), meta.to(device, non_blocking=True)
                with autocast():
                    logits = model(X, meta)
                    loss = criterion(logits, y)
                val_losses.append(loss.item())
                all_y.append(y.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
        all_y = np.concatenate(all_y, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        logger.log(f'VAL: logits shape={all_logits.shape}, dtype={all_logits.dtype}')
        logger.log(f'VAL: labels shape={all_y.shape}, dtype={all_y.dtype}, unique={np.unique(all_y)}')
        # Сохраняем массивы
        np.save(os.path.join(ARTIFACTS_DIR, f'val_y_epoch{epoch+1}.npy'), all_y)
        np.save(os.path.join(ARTIFACTS_DIR, f'val_logits_epoch{epoch+1}.npy'), all_logits)
        # --- Метрики только по threshold=0.3 ---
        probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
        preds_03 = (probs > 0.3).astype(int)
        macro_f1 = f1_score(all_y, preds_03, average='macro', zero_division=0)
        per_class_f1 = f1_score(all_y, preds_03, average=None, zero_division=0)
        logger.log(f'VAL: macro_f1@0.3={macro_f1:.4f}, per_class_f1={per_class_f1}')
        logger.log(f'VAL: classification_report@0.3:\n{classification_report(all_y, preds_03, zero_division=0)}')
        # Проверка: модель что-то предсказывает
        nonzero_pred_rate = (preds_03.sum(axis=1) > 0).mean()
        logger.log(f"[DEBUG] Процент сэмплов с хотя бы 1 предсказанным классом: {nonzero_pred_rate:.2%}")
        per_class_pred_rate = preds_03.mean(axis=0)
        logger.log(f"[DEBUG] Per-class pred rate: {per_class_pred_rate}")
        # --- optimize_thresholds только для анализа, не для метрик ---
        # thresholds = optimize_thresholds(all_y, probs)
        # np.save(os.path.join(ARTIFACTS_DIR, f'thresholds_epoch{epoch+1}.npy'), thresholds)
        # preds_opt = (probs > thresholds).astype(int)
        # macro_f1_opt = f1_score(all_y, preds_opt, average='macro', zero_division=0)
        # per_class_f1_opt = f1_score(all_y, preds_opt, average=None, zero_division=0)
        # logger.log(f'VAL: macro_f1 (opt thresholds)={macro_f1_opt:.4f}, per_class_f1={per_class_f1_opt}')
        # logger.log(f'VAL: classification_report (opt thresholds):\n{classification_report(all_y, preds_opt, zero_division=0)}')
        # Логирование "молчащих" классов
        for i, f1 in enumerate(per_class_f1):
            if f1 < 0.05:
                logger.log(f'[WARN] Class {i} почти не предсказывается: f1={f1:.3f}')
        # Coverage
        coverage = (per_class_f1 > 0.1).mean()
        logger.log(f'VAL: class coverage (F1 > 0.1): {coverage:.2%}')
        # Визуализация сигналов (первые 3)
        for i in range(3):
            fig, ax = plt.subplots()
            ax.plot(val_loader.dataset[i][0][0])
            fig.savefig(os.path.join(ARTIFACTS_DIR, f'val_signal_epoch{epoch+1}_sample{i}.png'))
            plt.close(fig)
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
            'config': CONFIG,
            'classes': frequent_classes,
        }
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
        # Сохраняем всегда last.pth
        save_checkpoint(state, CONFIG['checkpoint_dir'], is_best=False, filename='last.pth')
        # Сохраняем best.pth только если is_best
        if is_best:
            save_checkpoint(state, CONFIG['checkpoint_dir'], is_best=True, filename='best.pth')

        # --- Early stopping ---
        if patience >= CONFIG['early_stopping_patience']:
            print(f'Early stopping: val_f1 did not improve for {CONFIG["early_stopping_patience"]} epochs.')
            break

    print('Training completed!')

    # --- Testing after training ---
    ckpt_path = os.path.join(CONFIG['checkpoint_dir'], 'best.pth')
    if os.path.exists(ckpt_path):
        best_ckpt = load_checkpoint(CONFIG['checkpoint_dir'], filename='best.pth', map_location=CONFIG['device'])
        model.load_state_dict(best_ckpt['model_state_dict'])
        optimizer.load_state_dict(best_ckpt['optimizer_state_dict'])
        print(f'Loaded best checkpoint with val_f1 = {best_ckpt["val_f1"]:.4f} (epoch {best_ckpt["epoch"]})')

        model.eval()
        test_losses = []
        all_y = []
        all_logits = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test", leave=False):
                if batch is None:
                    continue
                X, y, meta = batch
                X, y, meta = X.to(device, non_blocking=True), y.to(device, non_blocking=True), meta.to(device, non_blocking=True)
                with autocast():
                    logits = model(X, meta)
                    loss = criterion(logits, y)
                test_losses.append(loss.item())
                all_y.append(y.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
        all_y = np.concatenate(all_y, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        # --- Test с оптимальными порогами ---
        thresholds_path = os.path.join(ARTIFACTS_DIR, f'thresholds_epoch{best_ckpt["epoch"]}.npy')
        if os.path.exists(thresholds_path):
            thresholds = np.load(thresholds_path)
            probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
            preds_opt = (probs > thresholds).astype(int)
            test_macro_f1_opt = f1_score(all_y, preds_opt, average='macro', zero_division=0)
            logger.log(f'TEST: macro F1 (opt thresholds) = {test_macro_f1_opt:.4f}')
        else:
            logger.log('[WARNING] Нет файла с оптимальными порогами для теста!')

        # Метрики только по threshold=0.3
        probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
        preds_03 = (probs > 0.3).astype(int)
        macro_f1 = f1_score(all_y, preds_03, average='macro', zero_division=0)
        per_class_f1 = f1_score(all_y, preds_03, average=None, zero_division=0)
        logger.log(f'TEST: macro_f1@0.3={macro_f1:.4f}, per_class_f1={per_class_f1}')
        logger.log(f'TEST: classification_report@0.3:\n{classification_report(all_y, preds_03, zero_division=0)}')
        nonzero_pred_rate = (preds_03.sum(axis=1) > 0).mean()
        logger.log(f"[DEBUG] TEST: Процент сэмплов с хотя бы 1 предсказанным классом: {nonzero_pred_rate:.2%}")
        per_class_pred_rate = preds_03.mean(axis=0)
        logger.log(f"[DEBUG] TEST: Per-class pred rate: {per_class_pred_rate}")
    else:
        logger.log('[WARNING] Нет best.pth для теста, тест не выполнен!')

    # График динамики F1 по эпохам
    metrics_path = os.path.join(LOGS_DIR, f'{EXPERIMENT_NAME}_metrics.json')
    if os.path.exists(metrics_path):
        metrics = json.load(open(metrics_path, encoding='utf-8'))
        f1s = [m['val_f1'] for m in metrics if isinstance(m['step'], int) and 'val_f1' in m]
        if f1s:
            plt.figure()
            plt.plot(f1s)
            plt.xlabel('Epoch')
            plt.ylabel('Val F1')
            plt.title('Val Macro F1 per Epoch')
            plt.savefig(os.path.join(ARTIFACTS_DIR, 'val_f1_curve.png'))
            plt.close()

if __name__ == '__main__':
    freeze_support()
    main() 
