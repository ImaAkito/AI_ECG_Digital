import pandas as pd
from ast import literal_eval
import numpy as np
import wfdb
import os
from collections import Counter
from src.augmentation.rare_class_generator import RareECGClassGenerator

# Parameters
PTBXL_CSV = 'data/raw/ptb-xl/ptbxl_database.csv'
RECORDS_DIR = 'data/raw/ptb-xl/records500/00000'
N_SYNTH = 100  # how many synthetic examples to generate per class
SAVE_DIR = 'data/synthetic_ecg/'
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Analyze dataset
print('Loading metadata...')
df = pd.read_csv(PTBXL_CSV)
df['scp_codes'] = df['scp_codes'].apply(literal_eval)
all_labels = []
for codes in df['scp_codes']:
    all_labels.extend(codes.keys())
label_counts = Counter(all_labels)
# Classes with 70 <= N < 500
target_classes = [k for k, v in label_counts.items() if 70 <= v < 500]
print('Synthetic ECGs will be generated for classes:', target_classes)

# 2. Build templates (averaged signals)
templates = {}
for cls in target_classes:
    rec_ids = df[df['scp_codes'].apply(lambda d: cls in d)].index
    signals = []
    for idx in rec_ids:
        rec_name = df.loc[idx, 'filename_hr'].split('/')[-1].replace('.dat', '')
        rec_path = os.path.join(RECORDS_DIR, rec_name)
        try:
            record = wfdb.rdrecord(rec_path)
            sig = record.p_signal.T  # (n_leads, siglen)
            signals.append(sig)
        except Exception as e:
            print(f'Error reading {rec_path}: {e}')
    if signals:
        template = np.mean(np.stack(signals), axis=0)
        templates[cls] = template
        print(f'Template for {cls}: {len(signals)} signals')
    else:
        print(f'No signals for {cls}')

# 3. Generate synthetic ECGs
if templates:
    generator = RareECGClassGenerator(templates)
    for cls in templates:
        synth_ecgs = generator.generate(cls, n=N_SYNTH)
        # Save numpy array
        np.save(os.path.join(SAVE_DIR, f'synthetic_{cls}.npy'), synth_ecgs)
        print(f'Saved {N_SYNTH} synthetic ECGs for class {cls}')
else:
    print('No templates for synthetic generation!') 