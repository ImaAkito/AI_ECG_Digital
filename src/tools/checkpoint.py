import torch
import os
from typing import Any, Dict, Optional

def save_checkpoint(state: Dict[str, Any], checkpoint_dir: str, is_best: bool = False, filename: str = 'last.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        torch.save(state, best_path)

def load_checkpoint(checkpoint_dir: str, filename: str = 'last.pth', map_location: Optional[str] = None) -> Dict[str, Any]:
    path = os.path.join(checkpoint_dir, filename)
    return torch.load(path, map_location=map_location, weights_only=False) 
