#Abstract base class for trainers, in order to replace the functions for classes in the training pipelines.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
import os as _os
import textwrap
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from cfg import CFG, DEFAULTS
from models.loss import setup_criterion
from utils.misc import ReprMixin, dict_to_str, dicts_equal, get_date_str, get_kwargs
from utils.utils_nn import default_collate_fn, make_safe_globals
from training.loggers import LoggerManager

from ..augmenters import AugmenterManager

__all__ = [
    "BaseTrainer",
]


class BaseTrainer(ReprMixin, ABC):
    """Abstract base class for trainers.

    A trainer is a class that contains the training pipeline,
    and is responsible for training a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained
    dataset_cls : torch.utils.data.Dataset
        The class of dataset to be used for training,
        `dataset_cls` should be inherited from :class:`~torch.utils.data.Dataset`,
        and be initialized via :code:`dataset_cls(config, training=True)`.
    model_config : dict
        The configuration of the model,
        used to keep a record in the checkpoints.
    train_config : dict
        The configuration of the training,
        including configurations for the data loader, for the optimization, etc.
        Will also be recorded in the checkpoints.
        `train_config` should at least contain the following keys:

            - "monitor": str
            - "loss": str
            - "n_epochs": int
            - "batch_size": int
            - "learning_rate": float
            - "lr_scheduler": str
                - "lr_step_size": int, optional, depending on the scheduler
                - "lr_gamma": float, optional, depending on the scheduler
                - "max_lr": float, optional, depending on the scheduler
            - "optimizer": str
                - "decay": float, optional, depending on the optimizer
                - "momentum": float, optional, depending on the optimizer
    collate_fn : callable, optional
        The collate function for the data loader,
        defaults to :meth:`default_collate_fn`.

        .. versionadded:: 0.0.23
    device : torch.device, optional
        The device to be used for training.
    lazy : bool, default False
        Whether to initialize the data loader lazily.

    """

    __name__ = "BaseTrainer"
    __DEFATULT_CONFIGS__ = {
        "debug": True,
        "final_model_name": None,
        "log_step": 10,
        "flooding_level": 0,
        "early_stopping": {},
    }
    __DEFATULT_CONFIGS__.update(deepcopy(DEFAULTS))

    def __init__(
        self,
        model: nn.Module,
        dataset_cls: Dataset,
        model_config: dict,
        train_config: dict,
        collate_fn: Optional[callable] = None,
        device: Optional[torch.device] = None,
        lazy: bool = False,
    ) -> None:
        self.model = model
        if type(self.model).__name__ in [
            "DataParallel",
        ]:
            # TODO: further consider "DistributedDataParallel"
            self._model = self.model.module
        else:
            self._model = self.model
        self.dataset_cls = dataset_cls
        self.model_config = CFG(deepcopy(model_config))
        self._train_config = CFG(deepcopy(train_config))
        self._train_config.checkpoints = Path(self._train_config.checkpoints)
        self.device = device or next(self._model.parameters()).device
        self.dtype = next(self._model.parameters()).dtype
        self.model.to(self.device)
        self.lazy = lazy
        self.collate_fn = collate_fn or default_collate_fn

        self.log_manager = None
        self.augmenter_manager = None
        self.train_loader = None
        self.val_train_loader = None
        self.val_loader = None
        self._setup_from_config(self._train_config)

        # monitor for training: challenge metric
        self.best_state_dict = OrderedDict()
        self.best_metric = -np.inf
        self.best_eval_res = dict()
        self.best_epoch = -1
        self.pseudo_best_epoch = -1

        self.saved_models = deque()
        self.model.train()
        self.global_step = 0
        self.epoch = 0
        self.epoch_loss = 0

    def train(self) -> OrderedDict:
        """Train the model.

        Returns
        -------
        best_state_dict : OrderedDict
            The state dict of the best model.

        """
        self._setup_optimizer()

        self._setup_scheduler()

        self._setup_criterion()

        if self.train_config.monitor is not None:
            # if monitor is set but val_loader is None, use train_loader for validation
            # and choose the best model based on the metrics on the train set
            if self.val_loader is None and self.val_train_loader is None:
                self.val_train_loader = self.train_loader
                self.log_manager.log_message(
                    (
                        "No separate validation set is provided, while monitor is set. "
                        "The training set will be used for validation, "
                        "and the best model will be selected based on the metrics on the training set"
                    ),
                    level=logging.WARNING,
                )

        msg = textwrap.dedent(
            f"""
            Starting training:
            ------------------
            Epochs:          {self.n_epochs}
            Batch size:      {self.batch_size}
            Learning rate:   {self.lr}
            Training size:   {self.n_train}
            Validation size: {self.n_val}
            Device:          {self.device.type}
            Optimizer:       {self.train_config.optimizer}
            Dataset classes: {self.train_config.classes}
            -----------------------------------------
            """
        )
        self.log_manager.log_message(msg)

        start_epoch = self.epoch
        for _ in range(start_epoch, self.n_epochs):
            # train one epoch
            self.model.train()
            self.epoch_loss = 0
            with tqdm(
                total=self.n_train,
                desc=f"Epoch {self.epoch}/{self.n_epochs}",
                unit="signals",
                dynamic_ncols=True,
                mininterval=1.0,
            ) as pbar:
                self.log_manager.epoch_start(self.epoch)
                # train one epoch
                self.train_one_epoch(pbar)

                # evaluate on train set, if debug is True
                if self.val_train_loader is not None:
                    eval_train_res = self.evaluate(self.val_train_loader)
                    self.log_manager.log_metrics(
                        metrics=eval_train_res,
                        step=self.global_step,
                        epoch=self.epoch,
                        part="train",
                    )
                else:
                    eval_train_res = {}
                # evaluate on val set
                if self.val_loader is not None:
                    eval_res = self.evaluate(self.val_loader)
                    self.log_manager.log_metrics(
                        metrics=eval_res,
                        step=self.global_step,
                        epoch=self.epoch,
                        part="val",
                    )
                elif self.val_train_loader is not None:
                    # if no separate val set, use the metrics on the train set
                    eval_res = eval_train_res
                else:
                    eval_res = {}

                # update best model and best metric if monitor is set
                if self.train_config.monitor is not None:
                    if eval_res[self.train_config.monitor] > self.best_metric:
                        self.best_metric = eval_res[self.train_config.monitor]
                        self.best_state_dict = self._model.state_dict()
                        self.best_eval_res = deepcopy(eval_res)
                        self.best_epoch = self.epoch
                        self.pseudo_best_epoch = self.epoch
                    elif self.train_config.early_stopping:
                        if eval_res[self.train_config.monitor] >= self.best_metric - self.train_config.early_stopping.min_delta:
                            self.pseudo_best_epoch = self.epoch
                        elif self.epoch - self.pseudo_best_epoch >= self.train_config.early_stopping.patience:
                            msg = f"early stopping is triggered at epoch {self.epoch}"
                            self.log_manager.log_message(msg)
                            break

                    msg = textwrap.dedent(
                        f"""
                        best metric = {self.best_metric},
                        obtained at epoch {self.best_epoch}
                    """
                    )
                    self.log_manager.log_message(msg)

                    # save checkpoint
                    save_suffix = f"epochloss_{self.epoch_loss:.5f}_metric_{eval_res[self.train_config.monitor]:.2f}"
                else:
                    save_suffix = f"epochloss_{self.epoch_loss:.5f}"
                save_filename = f"{self.save_prefix}_epoch{self.epoch}_{get_date_str()}_{save_suffix}.pth.tar"
                save_path = self.train_config.checkpoints / save_filename
                if self.train_config.keep_checkpoint_max != 0:
                    self.save_checkpoint(str(save_path))
                    self.saved_models.append(save_path)
                # remove outdated models
                if len(self.saved_models) > self.train_config.keep_checkpoint_max > 0:
                    model_to_remove = self.saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except Exception:
                        self.log_manager.log_message(f"failed to remove {str(model_to_remove)}")

                # update learning rate using lr_scheduler
                if self.train_config.lr_scheduler.lower() == "plateau":
                    self._update_lr(eval_res)

                self.log_manager.epoch_end(self.epoch)

            self.epoch += 1

        # save the best model
        if self.best_metric > -np.inf:
            if self.train_config.final_model_name:
                save_filename = self.train_config.final_model_name
            else:
                save_suffix = f"metric_{self.best_eval_res[self.train_config.monitor]:.2f}"
                save_filename = f"BestModel_{self.save_prefix}{self.best_epoch}_{get_date_str()}_{save_suffix}.pth.tar"
            save_path = self.train_config.model_dir / save_filename
            # self.save_checkpoint(path=str(save_path))
            self._model.save(path=str(save_path), train_config=self.train_config)
            self.log_manager.log_message(f"best model is saved at {save_path}")
        elif self.train_config.monitor is None:
            self.log_manager.log_message("no monitor is set, the last model is selected and saved as the best model")
            self.best_state_dict = self._model.state_dict()
            save_filename = f"BestModel_{self.save_prefix}{self.epoch}_{get_date_str()}.pth.tar"
            save_path = self.train_config.model_dir / save_filename
            # self.save_checkpoint(path=str(save_path))
            self._model.save(path=str(save_path), train_config=self.train_config)
        else:
            raise ValueError("No best model found!")

        self.log_manager.close()

        if not self.best_state_dict:
            # in case no best model is found,
            # e.g. monitor is not set, or keep_checkpoint_max is 0
            self.best_state_dict = self._model.state_dict()

        return self.best_state_dict

    def train_one_epoch(self, pbar: tqdm) -> None:
        """Train one epoch, and update the progress bar

        Parameters
        ----------
        pbar : tqdm
            The progress bar for training.

        """
        for epoch_step, data in enumerate(self.train_loader):
            self.global_step += 1
            # data is assumed to be a tuple of tensors, of the following order:
            # signals, labels, *extra_tensors
            data = self.augmenter_manager(*data)
            out_tensors = self.run_one_step(*data)

            loss = self.criterion(*out_tensors).to(self.dtype)
            if self.train_config.flooding_level > 0:
                flood = (loss - self.train_config.flooding_level).abs() + self.train_config.flooding_level
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                flood.backward()
            else:
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
            self.optimizer.step()
            self._update_lr()

            if self.global_step % self.train_config.log_step == 0:
                train_step_metrics = {"loss": loss.item()}
                if self.scheduler:
                    train_step_metrics.update({"lr": self.scheduler.get_last_lr()[0]})
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                            "lr": self.scheduler.get_last_lr()[0],
                        }
                    )
                else:
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                        }
                    )
                if self.train_config.flooding_level > 0:
                    train_step_metrics.update({"flood": flood.item()})
                self.log_manager.log_metrics(
                    metrics=train_step_metrics,
                    step=self.global_step,
                    epoch=self.epoch,
                    part="train",
                )
            pbar.update(data[0].shape[self.batch_dim])

    @property
    @abstractmethod
    def batch_dim(self) -> int:
        """The batch dimension

        Usually 0, but can be 1 for some models, e.g. :class:`~torch_ecg.models.RR_LSTM`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def extra_required_train_config_fields(self) -> List[str]:
        """Extra required fields in `train_config`."""
        raise NotImplementedError

    @property
    def required_train_config_fields(self) -> List[str]:
        """Required fields in `train_config`."""
        return [
            "classes",
            # "monitor",  # can be None
            "n_epochs",
            "batch_size",
            "log_step",
            "optimizer",
            "lr_scheduler",
            "learning_rate",
        ] + self.extra_required_train_config_fields

    def _validate_train_config(self) -> None:
        """Validate the `train_config`.

        Check if all required fields are present.
        """
        for field in self.required_train_config_fields:
            if field not in self.train_config:
                raise ValueError(f"{field} is missing in train_config!")

    @property
    def save_prefix(self) -> str:
        """The prefix of the saved model name."""
        model_name = self._model.__name__ if hasattr(self._model, "__name__") else self._model.__class__.__name__
        return f"{model_name}_epoch"

    @property
    def train_config(self) -> CFG:
        return self._train_config

    @abstractmethod
    def run_one_step(self, *data: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Run one step of training on one batch of data.

        Parameters
        ----------
        data : Tuple[torch.Tensor]
            The data to be processed for training one step (batch),
            should be of the following order:
            ``signals, labels, *extra_tensors``.

        Returns
        -------
        Tuple[torch.Tensor]
            The output of the model for one step (batch) data,
            along with labels and extra tensors.
            Should be of the following order:
            ``preds, labels, *extra_tensors``.
            `preds` usually are NOT the logits,
            but tensors before fed into :meth:`~torch.sigmoid`
            or :meth:`~torch.softmax` to get the logits.

        """
        raise NotImplementedError

    @torch.no_grad()
    @abstractmethod
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Do evaluation on the given data loader.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data loader to evaluate on.

        Returns
        -------
        dict
            The evaluation results (metrics).

        """
        raise NotImplementedError

    def _update_lr(self, eval_res: Optional[dict] = None) -> None:
        """Update learning rate using lr_scheduler,
        perhaps based on the `eval_res`.

        Parameters
        ----------
        eval_res : dict, optional
            The evaluation results (metrics).

        """
        if self.train_config.lr_scheduler.lower() == "none":
            pass
        elif self.train_config.lr_scheduler.lower() == "plateau":
            if eval_res is None:
                return
            metrics = eval_res[self.train_config.monitor]
            if isinstance(metrics, torch.Tensor):
                metrics = metrics.item()
            self.scheduler.step(metrics)
        elif self.train_config.lr_scheduler.lower() == "step":
            self.scheduler.step()
        elif self.train_config.lr_scheduler.lower() in [
            "one_cycle",
            "onecycle",
        ]:
            self.scheduler.step()

    def _setup_from_config(self, train_config: dict) -> None:
        """Setup the trainer from the training configuration.

        Parameters
        ----------
        train_config : dict
            The training configuration.

        """
        _default_config = CFG(deepcopy(self.__DEFATULT_CONFIGS__))
        _default_config.update(train_config)
        self._train_config = CFG(deepcopy(_default_config))

        # check validity of the config
        self._validate_train_config()

        # set aliases
        self.n_epochs = self.train_config.n_epochs
        self.batch_size = self.train_config.batch_size
        self.lr = self.train_config.learning_rate

        # setup log manager first
        self._setup_log_manager()
        msg = f"training configurations are as follows:\n{dict_to_str(self.train_config)}"
        self.log_manager.log_message(msg)

        # setup directories
        self._setup_directories()

        # setup callbacks
        self._setup_callbacks()

        # setup data loaders
        if not self.lazy:
            self._setup_dataloaders()

        # setup augmenters manager
        self._setup_augmenter_manager()

    def extra_log_suffix(self) -> str:
        """Extra suffix for the log file name."""
        model_name = self._model.__name__ if hasattr(self._model, "__name__") else self._model.__class__.__name__
        return f"{model_name}_{self.train_config.optimizer}_LR_{self.lr}_BS_{self.batch_size}"

    def _setup_log_manager(self) -> None:
        """Setup the log manager."""
        config = {"log_suffix": self.extra_log_suffix()}
        config.update(self.train_config)
        self.log_manager = LoggerManager.from_config(config=config)

    def _setup_directories(self) -> None:
        """Setup the directories for saving checkpoints and logs."""
        if not self.train_config.get("model_dir", None):
            self._train_config.model_dir = self.train_config.checkpoints
        self._train_config.model_dir = Path(self._train_config.model_dir)
        self.train_config.checkpoints.mkdir(parents=True, exist_ok=True)
        self.train_config.model_dir.mkdir(parents=True, exist_ok=True)

    def _setup_callbacks(self) -> None:
        """Setup the callbacks."""
        self._train_config.monitor = self.train_config.get("monitor", None)
        if self.train_config.monitor is None:
            assert (
                self.train_config.lr_scheduler.lower() != "plateau"
            ), "monitor is not specified, lr_scheduler should not be ReduceLROnPlateau"
        self._train_config.keep_checkpoint_max = self.train_config.get("keep_checkpoint_max", 1)
        if self._train_config.keep_checkpoint_max < 0:
            self._train_config.keep_checkpoint_max = -1
            self.log_manager.log_message(
                msg="keep_checkpoint_max is set to -1, all checkpoints will be kept",
                level=logging.WARNING,
            )
        elif self._train_config.keep_checkpoint_max == 0:
            self.log_manager.log_message(
                msg="keep_checkpoint_max is set to 0, no checkpoint will be kept",
                level=logging.WARNING,
            )

    def _setup_augmenter_manager(self) -> None:
        """Setup the augmenter manager."""
        self.augmenter_manager = AugmenterManager.from_config(config=self.train_config)

    @abstractmethod
    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        """Setup the dataloaders for training and validation.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset, optional
            The training dataset.
        val_dataset : torch.utils.data.Dataset, optional
            The validation dataset

        Examples
        --------
        .. code-block:: python

            if train_dataset is None:
                train_dataset = self.dataset_cls(config=self.train_config, training=True, lazy=False)
            if val_dataset is None:
                val_dataset = self.dataset_cls(config=self.train_config, training=False, lazy=False)
            num_workers = 4
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.collate_fn,
            )
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.collate_fn,
            )

        """
        raise NotImplementedError

    @property
    def n_train(self) -> int:
        if self.train_loader is not None:
            return len(self.train_loader.dataset)
        return 0

    @property
    def n_val(self) -> int:
        if self.val_loader is not None:
            return len(self.val_loader.dataset)
        return 0

    def _setup_optimizer(self) -> None:
        """Setup the optimizer."""
        if self.train_config.optimizer.lower() == "adam":
            optimizer_kwargs = get_kwargs(optim.Adam)
            optimizer_kwargs.update({k: self.train_config.get(k, v) for k, v in optimizer_kwargs.items()})
            optimizer_kwargs.update(dict(lr=self.lr))
            self.optimizer = optim.Adam(
                params=self.model.parameters(),
                **optimizer_kwargs,
            )
        elif self.train_config.optimizer.lower() in ["adamw", "adamw_amsgrad"]:
            optimizer_kwargs = get_kwargs(optim.AdamW)
            optimizer_kwargs.update({k: self.train_config.get(k, v) for k, v in optimizer_kwargs.items()})
            optimizer_kwargs.update(
                dict(
                    lr=self.lr,
                    amsgrad=self.train_config.optimizer.lower().endswith("amsgrad"),
                )
            )
            self.optimizer = optim.AdamW(
                params=self.model.parameters(),
                **optimizer_kwargs,
            )
        elif self.train_config.optimizer.lower() == "sgd":
            optimizer_kwargs = get_kwargs(optim.SGD)
            optimizer_kwargs.update({k: self.train_config.get(k, v) for k, v in optimizer_kwargs.items()})
            optimizer_kwargs.update(dict(lr=self.lr))
            self.optimizer = optim.SGD(
                params=self.model.parameters(),
                **optimizer_kwargs,
            )
        else:
            raise NotImplementedError(
                f"optimizer `{self.train_config.optimizer}` not implemented! "
                "Please use one of the following: `adam`, `adamw`, `adamw_amsgrad`, `sgd`, "
                "or override this method to setup your own optimizer."
            )

    def _setup_scheduler(self) -> None:
        """Setup the learning rate scheduler."""
        if self.train_config.lr_scheduler is None or self.train_config.lr_scheduler.lower() == "none":
            self.train_config.lr_scheduler = "none"
            self.scheduler = None
        elif self.train_config.lr_scheduler.lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                "max",
                patience=2,
                verbose=False,
            )
        elif self.train_config.lr_scheduler.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                self.train_config.lr_step_size,
                self.train_config.lr_gamma,
                # verbose=False,
            )
        elif self.train_config.lr_scheduler.lower() in [
            "one_cycle",
            "onecycle",
        ]:
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.train_config.max_lr,
                epochs=self.n_epochs,
                steps_per_epoch=len(self.train_loader),
                # verbose=False,
            )
        else:  # TODO: add linear and linear with warmup schedulers
            raise NotImplementedError(
                f"lr scheduler `{self.train_config.lr_scheduler.lower()}` not implemented for training! "
                "Please use one of the following: `none`, `plateau`, `step`, `one_cycle`, "
                "or override this method to setup your own lr scheduler."
            )

    def _setup_criterion(self) -> None:
        """Setup the loss function."""
        loss_kw = self.train_config.get("loss_kw", {})
        for k, v in loss_kw.items():
            if isinstance(v, torch.Tensor):
                loss_kw[k] = v.to(device=self.device, dtype=self.dtype)
        self.criterion = setup_criterion(self.train_config.loss, **loss_kw)
        self.criterion.to(self.device)

    def _check_model_config_compatability(self, model_config: dict) -> bool:
        """Check if `model_config` is compatible with the current model configuration.

        Parameters
        ----------
        model_config : dict
            Model configuration from elsewhere (e.g. from a checkpoint),
            which should be compatible with the current model configuration.

        Returns
        -------
        bool
            True if compatible, False otherwise

        """
        return dicts_equal(self.model_config, model_config)

    def resume_from_checkpoint(self, checkpoint: Union[str, dict]) -> None:
        """Resume a training process from a checkpoint.

        Parameters
        ----------
        checkpoint : str or dict
            If it is str, then it is the path of the checkpoint,
            which is a ``.pth.tar`` file containing a dict.
            `checkpoint` should contain at least
            "model_state_dict", "optimizer_state_dict",
            "model_config", "train_config", "epoch"
            to resume a training process.

        .. note::

            NOT finished, NOT tested.

        """
        if isinstance(checkpoint, str):
            ckpt = torch.load(checkpoint, map_location=self.device)
        else:
            ckpt = checkpoint
        insufficient_msg = "this checkpoint has no sufficient data to resume training"
        assert isinstance(ckpt, dict), insufficient_msg
        assert set(
            [
                "model_state_dict",
                "optimizer_state_dict",
                "model_config",
                "train_config",
                "epoch",
            ]
        ).issubset(ckpt.keys()), insufficient_msg
        if not self._check_model_config_compatability(ckpt["model_config"]):
            raise ValueError("model config of the checkpoint is not compatible with the config of the current model")
        self._model.load_state_dict(ckpt["model_state_dict"])
        self.epoch = ckpt["epoch"]
        self._setup_from_config(ckpt["train_config"])
        # TODO: resume optimizer, etc.

    def save_checkpoint(self, path: str) -> None:
        """Save the current state of the trainer to a checkpoint.

        Parameters
        ----------
        path : str
            Path to save the checkpoint

        """
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": make_safe_globals(self.model_config),
                "train_config": make_safe_globals(self.train_config),
                "epoch": self.epoch,
            },
            path,
        )

    def extra_repr_keys(self) -> List[str]:
        return [
            "train_config",
        ]
