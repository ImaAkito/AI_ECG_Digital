import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentLogger:
    def __init__(self, log_dir: str = './logs', exp_name: Optional[str] = None, to_console: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.exp_name = exp_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'{self.exp_name}.log')
        self.metrics_path = os.path.join(log_dir, f'{self.exp_name}_metrics.json')
        self.to_console = to_console
        self.metrics = {}
        with open(self.log_path, 'w') as f:
            f.write(f'Experiment: {self.exp_name}\n')
            f.write(f'Started: {datetime.now()}\n')

    def log(self, msg: str):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] {msg}'
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')
        if self.to_console:
            print(line)

    def log_params(self, params: Dict[str, Any]):
        self.log(f'PARAMS: {json.dumps(params, ensure_ascii=False)}')

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if step is not None:
            self.metrics[step] = metrics
        else:
            self.metrics[str(datetime.now())] = metrics
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        self.log(f'METRICS{f" (step {step})" if step is not None else ""}: {metrics}')

    def save_figure(self, fig, name: str):
        fig_path = os.path.join(self.log_dir, f'{self.exp_name}_{name}.png')
        fig.savefig(fig_path)
        self.log(f'Figure saved: {fig_path}') 