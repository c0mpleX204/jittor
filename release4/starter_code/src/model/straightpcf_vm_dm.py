from copy import deepcopy
from typing import Dict, List, Optional

import jittor as jt

from .dcvm import DirectionDistanceVelocityModule
from .feature import Decoder, FeatureExtraction
from .spec import ModelSpec
from .vm import patch_based_denoise

from ..data.asset import Asset


def _without_target(cfg):
    cfg = deepcopy(cfg)
    if "__target__" in cfg:
        del cfg["__target__"]
    return cfg


def _clip(x, min_value: Optional[float], max_value: Optional[float]):
    if min_value is not None:
        x = jt.maximum(x, jt.ones_like(x) * min_value)
    if max_value is not None:
        x = jt.minimum(x, jt.ones_like(x) * max_value)
    return x


class StraightPCFCoupledVelocityModule(ModelSpec):
    """
    Official-style StraightPCF coupled velocity stage.

    Compared with the earlier CoupledSurfaceStraightVelocityModule, this uses
    the patch interpolation time t. Each VM learns the same straight velocity,
    but is trained on a different segment from current t toward the clean
    endpoint.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.num_modules = cfg.get("num_modules", cfg.get("num_coupled_vms", 2))
        self.tot_its = cfg.get("tot_its", cfg.get("denoise_steps", 4))
        self.num_train_points = cfg.get("num_train_points", 128)
        self.dsm_sigma = cfg.get("dsm_sigma", 0.01)
        self.consistency_loss_weight = cfg.get("consistency_loss_weight", 10.0)

        velocity_cfg = cfg.get("velocity_model", None)
        if velocity_cfg is None:
            velocity_cfg = {
                "frame_knn": cfg.get("frame_knn", 16),
                "num_train_points": self.num_train_points,
                "feat_embedding_dim": cfg.get("feat_embedding_dim", 256),
                "decoder_hidden_dim": cfg.get("decoder_hidden_dim", 64),
                "dsm_sigma": self.dsm_sigma,
                "denoise_steps": self.tot_its,
            }
        velocity_cfg = _without_target(velocity_cfg)

        self.vm_names = []
        init_ckpt = cfg.get("init_ckpt", cfg.get("velocity_ckpt", None))
        for i in range(self.num_modules):
            name = f"velocity_model_{i}"
            vm = DirectionDistanceVelocityModule(
                model_config=velocity_cfg,
                transform_config=transform_config,
            )
            if init_ckpt is not None:
                vm.load(init_ckpt)
            setattr(self, name, vm)
            self.vm_names.append(name)

    def _vms(self):
        return [getattr(self, name) for name in self.vm_names]

    def _predict_velocity(self, vm, pc_state):
        B, N, d = pc_state.shape
        feat = vm.encoder(pc_state)
        F_dim = feat.shape[-1]
        return vm.decoder(c=feat.reshape(-1, F_dim)).reshape(B, N, d)

    def _time(self, pc_time, batch_size: int):
        if pc_time is None:
            return jt.zeros((batch_size, 1, 1))
        return _clip(pc_time.reshape(batch_size, 1, 1), 0.0, 1.0)

    def get_supervised_loss(self, pc_noisy_l2, pc_clean, pc_time=None):
        B, N, d = pc_noisy_l2.shape

        t = self._time(pc_time, B)
        remaining = 1.0 - t
        step_ratio = remaining / float(self.num_modules)
        target_velocity = pc_clean - pc_noisy_l2
        pc_state = pc_noisy_l2 + t.broadcast(pc_noisy_l2.shape) * target_velocity

        total_dir_loss = 0.0
        total_consistency_loss = 0.0

        for mod, vm in enumerate(self._vms()):
            pred_velocity = self._predict_velocity(vm, pc_state)
            dir_loss = ((pred_velocity - target_velocity) ** 2.0).sum(dim=-1).mean()
            total_dir_loss = total_dir_loss + dir_loss

            pc_state = pc_state + step_ratio.broadcast(pc_state.shape) * pred_velocity

            if mod < self.num_modules - 1:
                next_alpha = t + float(mod + 1) * step_ratio
                pc_expected = pc_noisy_l2 + next_alpha.broadcast(pc_noisy_l2.shape) * target_velocity
                consistency_loss = ((pc_expected - pc_state) ** 2.0).sum(dim=-1).mean()
                total_consistency_loss = total_consistency_loss + consistency_loss

        return (
            total_dir_loss +
            self.consistency_loss_weight * total_consistency_loss
        ) / self.dsm_sigma

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None):
        if num_steps is None:
            num_steps = self.tot_its

        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            denom = float(num_steps * self.num_modules)
            for _ in range(num_steps):
                for vm in self._vms():
                    pred_velocity = self._predict_velocity(vm, pcl_next)
                    pcl_next = pcl_next + (1.0 / denom) * pred_velocity
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        pc_noisy_l2 = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, patch_size, 3)
        pc_time = batch.get("pc_time", None)
        if pc_time is not None:
            pc_time = pc_time.reshape(-1)
        loss = self.get_supervised_loss(
            pc_noisy_l2=pc_noisy_l2,
            pc_clean=pc_clean,
            pc_time=pc_time,
        )
        return {"loss": loss}

    def execute(self, **kwargs) -> Dict:  # type: ignore
        return self.training_step(**kwargs)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch["pc_noisy"]
        assert pc_noisy_batch.ndim == 3

        res = []
        for pc_noisy in pc_noisy_batch:
            pc_next = patch_based_denoise(
                model=self,  # type: ignore[arg-type]
                pcl_noisy=pc_noisy,
                patch_size=1000,
                seed_k=6,
                seed_k_alpha=1,
            )
            pc_denoised = pc_next.detach().numpy()
            res.append({"pc_denoised": pc_denoised})
        return res

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                d = {
                    "pc_noisy": b.meta["pc_noisy"],
                    "pc_clean": b.meta["pc_clean"],
                }
                if "pc_time" in b.meta:
                    d["pc_time"] = b.meta["pc_time"]
                res.append(d)
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy,
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res


class StraightPCFVelocityDistanceModule(ModelSpec):
    """
    Official-style StraightPCF CVM + DistanceModule.

    The distance branch predicts one patch-level ratio. Training also runs the
    coupled VMs with that ratio and applies the StraightPCF reconstruction
    finetune loss.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.num_modules = cfg.get("num_modules", cfg.get("num_coupled_vms", 2))
        self.tot_its = cfg.get("tot_its", cfg.get("denoise_steps", 4))
        self.dsm_sigma = cfg.get("dsm_sigma", 0.01)
        self.ratio_min = cfg.get("ratio_min", 0.0)
        self.ratio_max = cfg.get("ratio_max", 1.0)
        self.target_ratio_min = cfg.get("target_ratio_min", self.ratio_min)
        self.target_ratio_max = cfg.get("target_ratio_max", self.ratio_max)
        self.ratio_loss_weight = cfg.get("ratio_loss_weight", 1.0)
        self.finetune_loss_weight = cfg.get("finetune_loss_weight", 200.0)
        self.freeze_velocity = cfg.get("freeze_velocity", False)
        self.velocity_eval_mode = cfg.get("velocity_eval_mode", True)
        self.recompute_ratio_each_iter = cfg.get("recompute_ratio_each_iter", False)

        coupled_cfg = cfg.get("coupled_model", None)
        if coupled_cfg is None:
            coupled_cfg = {
                "num_modules": self.num_modules,
                "tot_its": self.tot_its,
                "frame_knn": cfg.get("frame_knn", 16),
                "num_train_points": cfg.get("num_train_points", 128),
                "feat_embedding_dim": cfg.get("feat_embedding_dim", 256),
                "decoder_hidden_dim": cfg.get("decoder_hidden_dim", 64),
                "dsm_sigma": self.dsm_sigma,
                "velocity_model": cfg.get("velocity_model", None),
                "init_ckpt": cfg.get("init_ckpt", None),
            }
        coupled_cfg = _without_target(coupled_cfg)
        coupled_cfg["num_modules"] = coupled_cfg.get("num_modules", self.num_modules)
        coupled_cfg["tot_its"] = coupled_cfg.get("tot_its", self.tot_its)
        coupled_cfg["dsm_sigma"] = coupled_cfg.get("dsm_sigma", self.dsm_sigma)

        self.coupled_model = StraightPCFCoupledVelocityModule(
            model_config=coupled_cfg,
            transform_config=transform_config,
        )

        cvm_ckpt = cfg.get("cvm_ckpt", cfg.get("coupled_ckpt", None))
        if cvm_ckpt is not None:
            self.coupled_model.load(cvm_ckpt)

        distance_embedding_dim = cfg.get(
            "distance_feat_embedding_dim",
            cfg.get("feat_embedding_dim", 128),
        )
        self.distance_encoder = FeatureExtraction(
            k=cfg.get("distance_frame_knn", cfg.get("frame_knn", 16)),
            input_dim=3,
            embedding_dim=distance_embedding_dim,
            distance_estimation=cfg.get("distance_estimation", True),
        )
        self.distance_decoder = Decoder(
            z_dim=self.distance_encoder.embedding_dim,
            dim=3,
            out_dim=1,
            hidden_size=cfg.get("distance_decoder_hidden_dim", cfg.get("decoder_hidden_dim", 64)),
        )

    def _target_ratio(self, pc_noisy_l2, pc_current, pc_clean, pc_time=None):
        B = pc_noisy_l2.shape[0]
        if pc_time is not None:
            ratio = 1.0 - pc_time.reshape(B, 1, 1)
        else:
            num = jt.sqrt(((pc_clean - pc_current) ** 2.0).sum(dim=-1) + 1e-12)
            den = jt.sqrt(((pc_clean - pc_noisy_l2) ** 2.0).sum(dim=-1) + 1e-12)
            ratio = (num[:, 0] / (den[:, 0] + 1e-8)).reshape(B, 1, 1)
        return _clip(ratio, self.target_ratio_min, self.target_ratio_max)

    def predict_ratio(self, pc_current):
        B, N, _ = pc_current.shape
        feat = self.distance_encoder(pc_current)
        F_dim = feat.shape[-1]
        ratio01 = self.distance_decoder(
            c=feat.reshape(-1, F_dim),
            B=B,
            N=N,
        )
        return self.ratio_min + (self.ratio_max - self.ratio_min) * ratio01

    def _predict_velocity(self, vm, pc_state):
        if self.freeze_velocity:
            with jt.no_grad():
                return self.coupled_model._predict_velocity(vm, pc_state)
        return self.coupled_model._predict_velocity(vm, pc_state)

    def _apply_coupled_step(self, pc_current, ratio, train_mode: bool):
        pc_state = pc_current
        if train_mode:
            scale = ratio / float(self.num_modules)
            for vm in self.coupled_model._vms():
                pred_velocity = self._predict_velocity(vm, pc_state)
                pc_state = pc_state + scale.broadcast(pc_state.shape) * pred_velocity
            return pc_state

        pred_ratio = ratio
        denom = float(self.tot_its * self.num_modules)
        for _ in range(self.tot_its):
            if self.recompute_ratio_each_iter:
                pred_ratio = self.predict_ratio(pc_state)
            scale = pred_ratio / denom
            for vm in self.coupled_model._vms():
                pred_velocity = self._predict_velocity(vm, pc_state)
                pc_state = pc_state + scale.broadcast(pc_state.shape) * pred_velocity
        return pc_state

    def get_supervised_loss(self, pc_noisy_l2, pc_current, pc_clean, pc_time=None):
        if self.velocity_eval_mode:
            self.coupled_model.eval()

        target_ratio = self._target_ratio(
            pc_noisy_l2=pc_noisy_l2,
            pc_current=pc_current,
            pc_clean=pc_clean,
            pc_time=pc_time,
        )
        pred_ratio = self.predict_ratio(pc_current)

        ratio_loss = ((pred_ratio - target_ratio) ** 2.0).mean()
        pc_pred = self._apply_coupled_step(
            pc_current=pc_current,
            ratio=pred_ratio,
            train_mode=True,
        )
        finetune_loss = ((pc_clean - pc_pred) ** 2.0).sum(dim=-1).mean()

        return (
            self.ratio_loss_weight * ratio_loss +
            self.finetune_loss_weight * finetune_loss
        ) / self.dsm_sigma

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None):
        if num_steps is not None:
            old_steps = self.tot_its
            self.tot_its = num_steps
        else:
            old_steps = None

        with jt.no_grad():
            if self.velocity_eval_mode:
                self.coupled_model.eval()
            pred_ratio = self.predict_ratio(pcl_noisy)
            pcl_next = self._apply_coupled_step(
                pc_current=pcl_noisy,
                ratio=pred_ratio,
                train_mode=False,
            )

        if old_steps is not None:
            self.tot_its = old_steps
        return pcl_next, pred_ratio

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_mix"].shape[-2]
        pc_noisy_l2 = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_current = batch["pc_mix"].reshape(-1, patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, patch_size, 3)
        pc_time = batch.get("pc_time", None)
        if pc_time is not None:
            pc_time = pc_time.reshape(-1)
        loss = self.get_supervised_loss(
            pc_noisy_l2=pc_noisy_l2,
            pc_current=pc_current,
            pc_clean=pc_clean,
            pc_time=pc_time,
        )
        return {"loss": loss}

    def execute(self, **kwargs) -> Dict:  # type: ignore
        return self.training_step(**kwargs)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch["pc_noisy"]
        assert pc_noisy_batch.ndim == 3

        res = []
        for pc_noisy in pc_noisy_batch:
            pc_next = patch_based_denoise(
                model=self,  # type: ignore[arg-type]
                pcl_noisy=pc_noisy,
                patch_size=1000,
                seed_k=6,
                seed_k_alpha=1,
            )
            pc_denoised = pc_next.detach().numpy()
            res.append({"pc_denoised": pc_denoised})
        return res

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                d = {
                    "pc_noisy": b.meta["pc_noisy"],
                    "pc_clean": b.meta["pc_clean"],
                    "pc_mix": b.meta["pc_mix"],
                }
                if "pc_time" in b.meta:
                    d["pc_time"] = b.meta["pc_time"]
                res.append(d)
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy,
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res
