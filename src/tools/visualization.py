import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict, Any


def plot_ecg( #Plotting one ECG signal (n_leads, length)
    signal: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
    label: Optional[List[str]] = None,
    title: Optional[str] = None,
    lead_names: Optional[List[str]] = None,
    figsize=(12, 8),
    max_leads: int = 12,
    show: bool = True,
    savepath: Optional[str] = None,
):
    n_leads = min(signal.shape[0], max_leads)
    fig, axes = plt.subplots(n_leads, 1, sharex=True, figsize=figsize)
    if n_leads == 1:
        axes = [axes]
    for i in range(n_leads):
        axes[i].plot(signal[i], lw=1)
        axes[i].set_ylabel(lead_names[i] if lead_names else f'Lead {i+1}')
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (samples)')
    if title:
        fig.suptitle(title)
    if meta:
        meta_str = ', '.join(f'{k}: {v}' for k, v in meta.items())
        fig.text(0.01, 0.99, meta_str, va='top', ha='left', fontsize=10)
    if label:
        label_str = ', '.join(label)
        fig.text(0.99, 0.99, label_str, va='top', ha='right', fontsize=10, color='tab:red')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig)


def plot_batch( #Plotting a batch of ECG signals (n_samples, n_leads, length)
    batch_signals: np.ndarray,
    batch_labels: np.ndarray,
    batch_meta: Optional[List[Dict[str, Any]]] = None,
    class_list: Optional[List[str]] = None,
    n_samples: int = 8,
    lead_names: Optional[List[str]] = None,
    figsize=(16, 10),
    show: bool = True,
    savepath: Optional[str] = None,
):
    n = min(n_samples, batch_signals.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]
    for i in range(n):
        sig = batch_signals[i]
        label_idx = np.where(batch_labels[i] > 0)[0]
        labels = [class_list[j] for j in label_idx] if class_list is not None else [str(j) for j in label_idx]
        meta = batch_meta[i] if batch_meta is not None else None
        plot_ecg(sig, meta=meta, label=labels, title=f'Sample {i}', lead_names=lead_names, show=False)
        axes[i].imshow(np.zeros((1,1)), visible=False)  # for correct placement
        axes[i].set_axis_off()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig)


def compare_augmentations( #Comparing original and augmented ECG signals (n_leads, length)
    original: np.ndarray,
    augmented: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
    label: Optional[List[str]] = None,
    lead_names: Optional[List[str]] = None,
    figsize=(14, 8),
    show: bool = True,
    savepath: Optional[str] = None,
):
    n_leads = min(original.shape[0], 12)
    fig, axes = plt.subplots(n_leads, 2, sharex=True, figsize=figsize)
    for i in range(n_leads):
        axes[i, 0].plot(original[i], lw=1)
        axes[i, 0].set_ylabel(lead_names[i] if lead_names else f'Lead {i+1}')
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].plot(augmented[i], lw=1, color='tab:orange')
        axes[i, 1].set_title('Augmented' if i == 0 else '')
        axes[i, 1].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('Time (samples)')
    axes[-1, 1].set_xlabel('Time (samples)')
    if meta:
        meta_str = ', '.join(f'{k}: {v}' for k, v in meta.items())
        fig.text(0.01, 0.99, meta_str, va='top', ha='left', fontsize=10)
    if label:
        label_str = ', '.join(label)
        fig.text(0.99, 0.99, label_str, va='top', ha='right', fontsize=10, color='tab:red')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig) 