import os
import numpy as np
import pandas as pd
from ast import literal_eval
from collections import Counter
from tqdm import tqdm
from dataloading.ptbxl_advanced_dataset import PTBXLAdvancedDataset
from preprocessing.ecg_preprocessing_pipeline import ECGPreprocessingPipeline
from augmentation.ecg_augmentation_pipeline import ECGAugmentationPipeline
from tools.hierarchical_decoding import PTBXLSCPHierarchy

# --- CONFIG ---
RAW_CSV = 'data/raw/ptb-xl/ptbxl_database.csv'
RECORDS_ROOT = 'data/raw/ptb-xl'
SCP_STATEMENTS = 'data/raw/ptb-xl/scp_statements.csv'
CACHE_ROOT = 'data/processed_ptbxl'

PREPROC_CONFIG = {
    'orig_fs': 500,
    'target_fs': 500,
    'target_len': 5000,
    'filter_low': 0.5,
    'filter_high': 40.0,
    'filter_order': 4,
    'baseline_short': 0.2,
    'baseline_long': 0.6,
    'smooth_method': 'mean',
    'smooth_window': 11,
    'polarity_window_sec': 2.0,
    'normalizer_method': 'zscore',
    'gap_flat_thr': 1e-3,
    'quality_flat_thr': 1e-3,
    'quality_nan_thr': 0.01,
    'quality_outlier_thr': 5.0,
}

FREQ_THRESHOLD = 400

os.makedirs(CACHE_ROOT, exist_ok=True)

# --- Loading and filtering classes ---
df = pd.read_csv(RAW_CSV)
df['scp_codes'] = df['scp_codes'].apply(literal_eval)
all_labels = []
for codes in df['scp_codes']:
    all_labels.extend(codes.keys())
label_counts = Counter(all_labels)
frequent_classes = [k for k, v in label_counts.items() if v >= FREQ_THRESHOLD]
print('Частые классы:', frequent_classes)

# --- Patient-level split ---
patients = df['patient_id'].unique()
np.random.seed(42)
np.random.shuffle(patients)
n = len(patients)
train_patients = set(patients[:int(0.8*n)])
val_patients = set(patients[int(0.8*n):int(0.9*n)])
test_patients = set(patients[int(0.9*n):])

def get_split_df(split):
    if split == 'train':
        return df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
    elif split == 'val':
        return df[df['patient_id'].isin(val_patients)].reset_index(drop=True)
    elif split == 'test':
        return df[df['patient_id'].isin(test_patients)].reset_index(drop=True)
    else:
        raise ValueError('split должен быть train/val/test')

# --- Pipelines ---
preproc = ECGPreprocessingPipeline(**PREPROC_CONFIG)
aug = ECGAugmentationPipeline()

# --- Hierarchy for decoding ---
hier = PTBXLSCPHierarchy(SCP_STATEMENTS)

# --- Caching ---
def cache_split(split, use_aug=False):
    split_dir = os.path.join(CACHE_ROOT, split)
    os.makedirs(split_dir, exist_ok=True)
    split_df = get_split_df(split)
    records = []
    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f'Processing {split}'):
        # --- Loading signal ---
        if row['filename_hr']:
            rec_path = os.path.join(RECORDS_ROOT, row['filename_hr'].replace('.dat', ''))
        else:
            continue
        import wfdb
        record = wfdb.rdrecord(rec_path)
        sig = record.p_signal.T.astype(np.float32)  # (n_leads, siglen)
        # --- Preprocessing ---
        sig = preproc.process(sig)['signal']
        # --- Augmentation (only for train) ---
        if use_aug:
            sig = aug.augment(sig)
        # --- Saving ---
        rec_name = os.path.basename(row['filename_hr']).replace('.dat', '')
        out_path = os.path.join(split_dir, f'{rec_name}.npy')
        np.save(out_path, sig.astype(np.float32))
        # --- Labels ---
        multilabel = np.zeros(len(frequent_classes), dtype=np.float32)
        for label in row['scp_codes'].keys():
            if label in frequent_classes:
                multilabel[frequent_classes.index(label)] = 1.0
        # --- Metadata ---
        meta = {
            'sex': row['sex'],
            'age': row['age'],
            'heart_axis': row.get('heart_axis', '')
        }
        # --- Decoding ---
        codes = [k for k in row['scp_codes'].keys() if k in frequent_classes]
        hierarchy = hier.decode(codes)
        records.append({
            'signal_path': out_path,
            'sex': meta['sex'],
            'age': meta['age'],
            'heart_axis': meta['heart_axis'],
            'labels': multilabel.tolist(),
            'scp_codes': codes,
            'hierarchy': hierarchy
        })
    # --- Saving CSV ---
    out_csv = os.path.join(CACHE_ROOT, f'{split}.csv')
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f'{split} ready! Signals: {len(records)}')

if __name__ == '__main__':
    # Delete old cache
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(CACHE_ROOT, split)
        if os.path.exists(split_dir):
            for f in os.listdir(split_dir):
                os.remove(os.path.join(split_dir, f))
    # Caching
    cache_split('train', use_aug=True)
    cache_split('val', use_aug=False)
    cache_split('test', use_aug=False)
    print('All done!') 