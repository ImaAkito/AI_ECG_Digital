import pandas as pd
from ast import literal_eval
import numpy as np
import wfdb
from torch.utils.data import Dataset
from typing import List, Optional, Callable, Dict
import os

class PTBXLFrequentDataset(Dataset): #Dataset for PTB-XL, including multi-label
    def __init__(self,
                 csv_path: str,
                 records_dir: str,
                 allowed_classes: List[str],
                 split: Optional[str] = None,  # 'train', 'val', 'test' or None (all)
                 split_col: str = 'strat_fold',
                 train_folds: List[int] = list(range(1,9)),
                 val_fold: int = 9,
                 test_fold: int = 10,
                 transform: Optional[Callable] = None,
                 max_len: int = 5000):
        self.df = pd.read_csv(csv_path)
        self.df['scp_codes'] = self.df['scp_codes'].apply(literal_eval)
        self.allowed_classes = allowed_classes
        self.transform = transform
        self.records_dir = records_dir
        self.max_len = max_len
        self.class_list = sorted(list(allowed_classes))
        self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}

        # Split
        if split is not None:
            if split == 'train':
                folds = train_folds
            elif split == 'val':
                folds = [val_fold]
            elif split == 'test':
                folds = [test_fold]
            else:
                raise ValueError('split must be train/val/test/None')
            self.df = self.df[self.df[split_col].isin(folds)]

        # Filter only needed records
        self.df = self.df[self.df['scp_codes'].apply(self._has_allowed_label)].reset_index(drop=True)

    def _has_allowed_label(self, codes: Dict[str, float]) -> bool:
        return any(label in self.allowed_classes for label in codes.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load signal
        rec_name = row['filename_hr'].split('/')[-1].replace('.dat', '')
        rec_path = os.path.join(self.records_dir, rec_name)
        record = wfdb.rdrecord(rec_path)
        sig = record.p_signal.T  # (n_leads, siglen)
        sig = self._fix_length(sig)
        # Form multi-label vector
        y = np.zeros(len(self.class_list), dtype=np.float32)
        for label in row['scp_codes'].keys():
            if label in self.class_to_idx:
                y[self.class_to_idx[label]] = 1.0
        # Augmentations/transformations
        if self.transform:
            sig = self.transform(sig)
        return sig, y

    def _fix_length(self, sig):
        # Truncation/padding to self.max_len
        if sig.shape[1] > self.max_len:
            start = (sig.shape[1] - self.max_len) // 2
            sig = sig[:, start:start+self.max_len]
        elif sig.shape[1] < self.max_len:
            pad = self.max_len - sig.shape[1]
            sig = np.pad(sig, ((0,0),(pad//2, pad - pad//2)), mode='constant')
        return sig

# Example integration with the training pipeline
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # 1. Get list of frequent classes
    df = pd.read_csv('data/raw/ptb-xl/ptbxl_database.csv')
    df['scp_codes'] = df['scp_codes'].apply(literal_eval)
    from collections import Counter
    all_labels = []
    for codes in df['scp_codes']:
        all_labels.extend(codes.keys())
    label_counts = Counter(all_labels)
    frequent_classes = [k for k, v in label_counts.items() if v >= 400]
    print('Frequent classes:', frequent_classes)
    # 2. Create datasets
    train_ds = PTBXLFrequentDataset(
        csv_path='data/raw/ptb-xl/ptbxl_database.csv',
        records_dir='data/raw/ptb-xl/records500/00000',
        allowed_classes=frequent_classes,
        split='train',
        transform=None
    )
    val_ds = PTBXLFrequentDataset(
        csv_path='data/raw/ptb-xl/ptbxl_database.csv',
        records_dir='data/raw/ptb-xl/records500/00000',
        allowed_classes=frequent_classes,
        split='val',
        transform=None
    )
    # 3. DataLoader
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    # 4. Example iteration
    for X, y in train_loader:
        print('Batch X:', X.shape, 'Batch y:', y.shape)
        break 