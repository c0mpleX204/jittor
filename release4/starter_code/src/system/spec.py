from collections import defaultdict
from jittor import optim
from typing import Dict, List, Optional
from tqdm import tqdm

import jittor as jt
import numpy as np
import os
import random

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule
from ..model.spec import ModelSpec

def _get_item(x):
    if isinstance(x, jt.Var):
        return x.item()
    return x

def get_optimizer(optimizer_config, model):
    optimizer_config = dict(optimizer_config)
    __target__ = optimizer_config.pop('__target__')
    MAPPING = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
    }
    if __target__ not in MAPPING:
        raise ValueError(f"unsupported optimizer: {__target__}")
    OptimizerClass = MAPPING[__target__]
    optimizer = OptimizerClass(model.parameters(), **optimizer_config)
    return optimizer

def _to_numpy_state(x):
    if isinstance(x, jt.Var):
        return x.numpy()
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, dict):
        return {k: _to_numpy_state(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_numpy_state(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_numpy_state(v) for v in x)
    return x

def optimizer_state_dict(optimizer) -> Dict:
    state = optimizer.state_dict()
    defaults = dict(state.get("defaults", {}))
    param_groups = defaults.get("param_groups", None)
    if param_groups is not None:
        cleaned_groups = []
        for group in param_groups:
            cleaned_groups.append({
                k: v for k, v in group.items()
                if k not in ("params", "grads")
            })
        defaults["param_groups"] = cleaned_groups
    return {"defaults": defaults}

def load_checkpoint(path: str) -> Dict:
    checkpoint = jt.load(path)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must be a dict: {path}")
    return checkpoint

def is_full_training_checkpoint(checkpoint: Dict) -> bool:
    return checkpoint.get("format") == "full_training_checkpoint_v1" and "model" in checkpoint

def load_model_state(model: ModelSpec, checkpoint_path: str) -> Dict:
    checkpoint = load_checkpoint(checkpoint_path)
    model_state = checkpoint["model"] if is_full_training_checkpoint(checkpoint) else checkpoint
    model.load_state_dict(model_state)
    if is_full_training_checkpoint(checkpoint):
        print(
            f"Loaded model weights from full checkpoint: {checkpoint_path} "
            f"(epoch={checkpoint.get('epoch')}, next_epoch={checkpoint.get('next_epoch')})"
        )
    else:
        print(f"Loaded model weights from legacy checkpoint: {checkpoint_path}")
    return checkpoint

class DummyWriter():
    
    def __init__(self):
        pass
    
    def write(self, batch, prediction: List[Dict], dataset_module: Optional[PCDatasetModule]=None):
        pass

class DummySystem():
    
    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        if trainer_config is None:
            trainer_config = {}
        self._start_epoch_from_config = 'start_epoch' in trainer_config
        self.start_epoch = trainer_config.get('start_epoch', 0)
        self.epochs = trainer_config.get('epochs', 1)
        
        if optimizer_config is not None and model is not None:
            self.optimizer = get_optimizer(optimizer_config, model)
        else:
            self.optimizer = None
        
        self._validation_loss = defaultdict(list)

    def checkpoint_state(self, epoch: int, loss=None) -> Dict:
        optimizer_state = None
        if self.optimizer is not None:
            optimizer_state = optimizer_state_dict(self.optimizer)

        return {
            "format": "full_training_checkpoint_v1",
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "model": self.model.state_dict(),
            "optimizer": optimizer_state,
            "loss": _get_item(loss) if loss is not None else None,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
        }

    def save_training_checkpoint(self, path: str, epoch: int, loss=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = _to_numpy_state(self.checkpoint_state(epoch=epoch, loss=loss))
        jt.save(checkpoint, path)
        print(f"Saved full training checkpoint: {path}")

    def load_training_checkpoint(self, path: str):
        checkpoint = load_model_state(self.model, path)
        if not is_full_training_checkpoint(checkpoint):
            raise ValueError(
                "resume_ckpt requires a full training checkpoint with model and optimizer state. "
                f"This looks like a legacy weights-only checkpoint: {path}. "
                "Use load_ckpt for warm start, or resume from a checkpoint saved after this patch."
            )

        if self.optimizer is None:
            raise ValueError("optimizer is None, cannot restore optimizer state")
        optimizer_state = checkpoint.get("optimizer", None)
        if optimizer_state is None:
            raise ValueError(f"checkpoint has no optimizer state: {path}")
        self.optimizer.load_state_dict(optimizer_state)

        rng_state = checkpoint.get("rng_state", {})
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])

        next_epoch = int(checkpoint.get("next_epoch", checkpoint.get("epoch", -1) + 1))
        if self._start_epoch_from_config and self.start_epoch != next_epoch:
            print(
                f"Warning: trainer.start_epoch={self.start_epoch} overrides "
                f"checkpoint next_epoch={next_epoch}"
            )
        elif not self._start_epoch_from_config:
            self.start_epoch = next_epoch

        print(
            f"Resumed training checkpoint: {path}; "
            f"epoch={checkpoint.get('epoch')}, next_epoch={self.start_epoch}, "
            f"optimizer_state=loaded"
        )
    
    def forward(self, batch, validate: bool=False): # return loss sum
        loss_dict = self.model.training_step(batch)
        assert isinstance(loss_dict, dict), "loss_dict must be a dict containing loss/metrics"
        assert self.loss_config is not None, "do not have loss_confing"
        loss_sum = 0.
        if validate:
            assets: List[Asset] = [a for a in batch['asset']]
            cls = assets[0].cls # guaranteed to be the same cls in dataloader
            for name in loss_dict:
                assert name in self.loss_config, f'unspecified loss {name}'
                self._validation_loss[f"val/{cls}_{name}"].append(_get_item(loss_dict[name]))
                loss_sum += self.loss_config[name] * loss_dict[name]
            self._validation_loss[f"val/{cls}_loss_sum"].append(_get_item(loss_sum))
            # TODO: log
            # self.log('val/loss_sum', loss_sum, prog_bar=True, logger=True, sync_dist=True, batch_size=len(assets))
        else:
            for name in loss_dict:
                assert name in self.loss_config, f"unspecified loss name: `{name}`"
                if self.loss_config[name] > 0:
                    loss_sum += self.loss_config[name] * loss_dict[name]
            loss_dict['loss_sum'] = loss_sum
            # TODO: log
            # # add train prefix to loss_dict
            # prefixed_loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            # d = dict(sorted(prefixed_loss_dict.items()))
        if not isinstance(loss_sum, jt.Var):
            return jt.array(loss_sum)
        return loss_sum
    
    def on_train_epoch_start(self):
        pass
    
    def on_train_batch_start(self):
        pass
    
    def training_step(self, batch):
        return self.forward(batch, validate=False)
    
    def on_train_batch_end(self):
        pass
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        self._validation_loss = defaultdict(list)
    
    def on_validation_batch_start(self):
        pass
    
    def validation_step(self, batch):
        assert self.loss_config is not None, "do not have loss_confing"
        return self.forward(batch, validate=True)
    
    def on_validation_batch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        pass
    
    def on_before_optimizer_step(self, optimizer):
        pass
    
    def on_predict_epoch_start(self):
        pass
    
    def on_predict_batch_start(self):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.model.predict_step(batch)
    
    def on_predict_batch_end(self):
        pass
    
    def on_predict_epoch_end(self):
        pass
    
    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.model.train()
            self.on_train_epoch_start()
            train_dataloader = self.dataset_module.train_dataloader()
            assert train_dataloader is not None, "train_dataloader is None"
            pbar = tqdm(train_dataloader, total=len(train_dataloader)//train_dataloader.batch_size) # type: ignore
            for batch in pbar:
                self.on_train_batch_start()
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                pbar.set_description(f"Epoch {epoch}, Loss: {_get_item(loss)}")
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self.on_train_batch_end()
            self.on_train_epoch_end()
            
            self.model.eval()
            validate_dataloader = self.dataset_module.validate_dataloader()
            if validate_dataloader is not None:
                self.on_validation_epoch_start()
                if isinstance(validate_dataloader, dict):
                    for name, dataloader in validate_dataloader.items():
                        pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size)
                        for batch in pbar:
                            self.on_validation_batch_start()
                            loss = self.validation_step(batch)
                            pbar.set_description(f"Epoch {epoch}, Validate {name}, Loss: {_get_item(loss)}")
                            self.on_validation_batch_end()
                else:
                    pbar = tqdm(validate_dataloader, total=len(validate_dataloader)//validate_dataloader.batch_size)
                    for batch in pbar:
                        self.on_validation_batch_start()
                        loss = self.validation_step(batch)
                        pbar.set_description(f"Epoch {epoch}, Validate, Loss: {_get_item(loss)}")
                        self.on_validation_batch_end()
                self.on_validation_epoch_end()
            
            checkpoint_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_{epoch}.pkl')
            self.save_training_checkpoint(checkpoint_path, epoch=epoch, loss=loss)
    
    def predict(self):
        # only iterate once
        self.model.set_predict(True)
        self.model.eval()
        self.on_predict_epoch_start()
        predict_dataloader = self.dataset_module.predict_dataloader()
        assert predict_dataloader is not None, "predict_dataloader is None"
        if not isinstance(predict_dataloader, dict):
            predict_dataloader = {"predict": predict_dataloader}
        for dataloader_name, dataloader in predict_dataloader.items():
            pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size) # type: ignore
            for batch_idx, batch in enumerate(pbar):
                self.on_predict_batch_start()
                output = self.predict_step(batch, batch_idx)
                if self.writer is not None:
                    self.writer.write(batch, output, dataset_module=self.dataset_module)
                pbar.set_description(f"Predicting {dataloader_name}, Batch {batch_idx}")
