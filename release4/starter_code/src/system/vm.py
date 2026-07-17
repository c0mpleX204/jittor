from typing import List, Dict, Optional

import numpy as np
import os

from .spec import DummySystem, DummyWriter
from ..data.asset import Asset, Exporter

class VMWriter(DummyWriter):
    
    def __init__(self, save_dir: str="tmp_predict", save_name: str="predict", output_format: str="npy"):
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name
        self.output_format = output_format

    def _relative_output_dir(self, path: str, dataset_module=None) -> str:
        if not os.path.isabs(path):
            return os.path.dirname(path)

        roots = []
        if dataset_module is not None and getattr(dataset_module, "predict_datapath", None) is not None:
            for datapath in dataset_module.predict_datapath.values():
                root = getattr(datapath, "input_dataset_dir", "")
                if root:
                    roots.append(os.path.abspath(root))

        abs_path = os.path.abspath(path)
        for root in roots:
            try:
                rel = os.path.relpath(abs_path, root)
            except ValueError:
                continue
            if not rel.startswith(".."):
                return os.path.dirname(rel)

        return os.path.dirname(os.path.basename(path))
    
    def write(self, batch, prediction: List[Dict], dataset_module=None):
        pc_noisy_batch = batch['pc_noisy']
        for i, asset in enumerate(batch['asset']):
            path = asset.path
            assert path is not None, "asset path is None"
            dirname = os.path.join(self.save_dir, self._relative_output_dir(path, dataset_module))
            os.makedirs(dirname, exist_ok=True)
            denoised = prediction[i]['pc_denoised']
            if isinstance(denoised, np.ndarray):
                denoised_np = denoised
            else:
                denoised_np = denoised.numpy()
            if self.output_format == 'npy':
                np.save(os.path.join(dirname, f"{self.save_name}.npy"), denoised_np.astype(np.float32))
            else:
                Exporter.export_obj(denoised_np, os.path.join(dirname, f"{self.save_name}.obj"))

class VMSystem(DummySystem):
    
    def __init__(
        self,
        dataset_module,
        model,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        super().__init__(
            dataset_module=dataset_module,
            model=model,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            writer=writer,
            ckpt_save_dir=ckpt_save_dir,
            ckpt_save_name=ckpt_save_name,
        )
    
    # override functions in dummy system if you want to implement training/validation/prediction logic
