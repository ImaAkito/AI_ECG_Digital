import pandas as pd
from ast import literal_eval
import numpy as np
import wfdb
from torch.utils.data import Dataset
from typing import List, Optional, Callable, Dict
import os

class PTBXLAdvancedDataset(Dataset): #Advanced dataset for PTB-XL, including meta-data, multi-label, and signal format
    def __init__(self,
                 csv_path: str,
                 records_dir: str = None,
                 allowed_classes: List[str] = None,
                 split: Optional[str] = None,  # 'train', 'val', 'test' or None (all)
                 split_col: str = 'strat_fold',
                 train_folds: List[int] = list(range(1,9)),
                 val_fold: int = 9,
                 test_fold: int = 10,
                 patient_split: bool = False,
                 signal_format: str = 'highres',  # 'highres', 'lowres', 'median'
                 transform: Optional[Callable] = None,
                 max_len: int = 5000,
                 meta_fields: Optional[List[str]] = None,
                 meta_transform: Optional[Callable] = None,
                 cache_mode: bool = False,
                 project_root: str = None):
        self.cache_mode = cache_mode
        self.transform = transform
        self.meta_transform = meta_transform
        self.max_len = max_len
        self.signal_format = signal_format
        self.records_dir = records_dir
        self.meta_fields = meta_fields or ['age', 'sex', 'patient_id']
        self.project_root = project_root or os.getcwd()

        if self.cache_mode:
            # Cached mode: reading CSV with paths to npy, labels and meta-data
            self.df = pd.read_csv(csv_path)
            self.class_list = None  # not used
            self.class_to_idx = None
        else:
            self.df = pd.read_csv(csv_path)
            self.df['scp_codes'] = self.df['scp_codes'].apply(literal_eval)
            self.allowed_classes = allowed_classes
            self.class_list = sorted(list(allowed_classes))
            self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}
            # Split by strat_fold or patient_id
            if patient_split:
                patients = self.df['patient_id'].unique()
                np.random.seed(42)
                np.random.shuffle(patients)
                n = len(patients)
                train_patients = set(patients[:int(0.8*n)])
                val_patients = set(patients[int(0.8*n):int(0.9*n)])
                test_patients = set(patients[int(0.9*n):])
                if split == 'train':
                    self.df = self.df[self.df['patient_id'].isin(train_patients)]
                elif split == 'val':
                    self.df = self.df[self.df['patient_id'].isin(val_patients)]
                elif split == 'test':
                    self.df = self.df[self.df['patient_id'].isin(test_patients)]
            elif split is not None:
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

    def _process_meta(self, row):
        # Age: normalization [0, 1] (max 100, everything above — 1.0)
        age = float(row.get('age', 0))
        age_norm = min(age / 100.0, 1.0)
        # Sex: binarization (M=1, F=0, everything else — -1)
        sex = row.get('sex', '')
        sex_bin = 1.0 if str(sex).upper() == 'M' else (0.0 if str(sex).upper() == 'F' else -1.0)
        # Heart axis: one-hot (6 variants, if not — all zeros)
        axis_classes = ['N', 'L', 'R', 'I', 'S', 'E']
        axis = str(row.get('heart_axis', '')).upper()
        axis_onehot = [1.0 if axis == c else 0.0 for c in axis_classes]
        # meta: [age_norm, sex_bin, *axis_onehot] (1+1+6=8)
        return np.array([age_norm, sex_bin] + axis_onehot, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.cache_mode:
            sig_path = row['signal_path']
            if not os.path.isabs(sig_path):
                sig_path = os.path.join(self.project_root, sig_path)
            sig = np.load(sig_path).astype(np.float32)
            y = np.array(eval(row['labels']), dtype=np.float32)
            meta = self._process_meta(row)
            return sig, y, meta
        else:
            if self.signal_format == 'highres':
                rec_name = row['filename_hr'].split('/')[-1].replace('.dat', '')
            elif self.signal_format == 'lowres':
                rec_name = row['filename_lr'].split('/')[-1].replace('.dat', '')
            elif self.signal_format == 'median' and 'median_filename' in row:
                rec_name = row['median_filename'].split('/')[-1].replace('.dat', '')
            else:
                raise ValueError('Unknown signal format or no median_filename')
            rec_path = os.path.join(self.records_dir, rec_name)
            record = wfdb.rdrecord(rec_path)
            sig = record.p_signal.T  # (n_leads, siglen)
            sig = self._fix_length(sig)
            meta = self._process_meta(row)
            y = np.zeros(len(self.class_list), dtype=np.float32)
            for label in row['scp_codes'].keys():
                if label in self.class_to_idx:
                    y[self.class_to_idx[label]] = 1.0
            if self.transform:
                sig = self.transform(sig, meta=meta)
            return sig, y, meta

    def _fix_length(self, sig):
        # Truncation/padding to self.max_len
        if sig.shape[1] > self.max_len:
            start = (sig.shape[1] - self.max_len) // 2
            sig = sig[:, start:start+self.max_len]
        elif sig.shape[1] < self.max_len:
            pad = self.max_len - sig.shape[1]
            sig = np.pad(sig, ((0,0),(pad//2, pad - pad//2)), mode='constant')
        return sig

# Example usage
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    df = pd.read_csv('data/raw/ptb-xl/ptbxl_database.csv')
    df['scp_codes'] = df['scp_codes'].apply(literal_eval)
    from collections import Counter
    all_labels = []
    for codes in df['scp_codes']:
        all_labels.extend(codes.keys())
    label_counts = Counter(all_labels)
    frequent_classes = [k for k, v in label_counts.items() if v >= 400]
    print('Frequent classes:', frequent_classes)
    # patient-level split, highres
    train_ds = PTBXLAdvancedDataset(
        csv_path='data/raw/ptb-xl/ptbxl_database.csv',
        records_dir='data/raw/ptb-xl/records500/00000',
        allowed_classes=frequent_classes,
        split='train',
        patient_split=True,
        signal_format='highres',
        transform=None
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    for X, y, meta in train_loader:
        print('Batch X:', X.shape, 'Batch y:', y.shape, 'Batch meta:', meta)
        break 