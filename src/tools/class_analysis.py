import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from collections import Counter

def plot_class_distribution(labels: np.ndarray, class_list: List[str], title: str = 'Class distribution', show: bool = True, savepath: Optional[str] = None): #Plotting the distribution of classes
    counts = labels.sum(axis=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(class_list)), counts)
    ax.set_xticks(range(len(class_list)))
    ax.set_xticklabels(class_list, rotation=90)
    ax.set_ylabel('Count')
    ax.set_title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig)

def plot_metadata_distribution(meta: List[Dict[str, Any]], field: str, bins: int = 20, show: bool = True, savepath: Optional[str] = None): #Plotting the distribution of metadata (e.g. age, patient_id)
    values = [m[field] for m in meta if field in m]
    fig, ax = plt.subplots(figsize=(8, 4))
    if isinstance(values[0], (int, float)):
        ax.hist(values, bins=bins, color='tab:blue', alpha=0.7)
    else:
        counts = Counter(values)
        ax.bar(counts.keys(), counts.values(), color='tab:blue', alpha=0.7)
        ax.set_xticklabels(counts.keys(), rotation=45)
    ax.set_title(f'Distribution of {field}')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig)

def print_class_stats(labels: np.ndarray, class_list: List[str]):
    counts = labels.sum(axis=0)
    total = labels.shape[0]
    print('Class statistics:')
    for i, c in enumerate(class_list):
        print(f'{c:20s}: {int(counts[i])} ({counts[i]/total*100:.2f}%)')
    print(f'Total samples: {total}') 