from copy import deepcopy
from typing import Dict, List, Optional

import jittor as jt
import numpy as np

from .feature import Decoder, FeatureExtraction
from .spec import ModelSpec
from .straightpcf_vm_dm import StraightPCFVelocityDistanceModule
from .vm import patch_based_denoise

from ..data.asset import Asset


def _without_target(cfg):
    cfg = deepcopy(cfg)
    if "__target__" in cfg:
        target = cfg["__target__"]
        if target != "StraightPCFVelocityDistanceModule":
            raise ValueError(f"CDRefine stage1_model expects StraightPCFVelocityDistanceModule, found {target}")
        del cfg["__target__"]
    return cfg


def _default_stage1_config() -> Dict:
    return {
        "num_modules": 2,
        "tot_its": 4,
        "dsm_sigma": 0.01,
        "frame_knn": 16,
        "feat_embedding_dim": 256,
        "decoder_hidden_dim": 64,
        "ratio_min": 0.0,
        "ratio_max": 1.0,
        "target_ratio_min": 0.0,
        "target_ratio_max": 1.0,
        "ratio_loss_weight": 1.0,
        "finetune_loss_weight": 200.0,
        "velocity_model": {
            "frame_knn": 16,
            "num_train_points": 128,
            "feat_embedding_dim": 256,
            "decoder_hidden_dim": 64,
            "dsm_sigma": 0.01,
            "denoise_steps": 4,
        },
        "coupled_model": {
            "num_modules": 2,
            "tot_its": 4,
            "num_train_points": 128,
            "dsm_sigma": 0.01,
            "consistency_loss_weight": 10.0,
            "velocity_model": {
                "frame_knn": 16,
                "num_train_points": 128,
                "feat_embedding_dim": 256,
                "decoder_hidden_dim": 64,
                "dsm_sigma": 0.01,
                "denoise_steps": 4,
            },
        },
    }


def _random_indices(n: int, m: Optional[int]):
    if m is None or m <= 0 or m >= n:
        return None
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()


def _sample_points(pc, num_points: Optional[int]):
    idx = _random_indices(pc.shape[1], num_points)
    if idx is None:
        return pc
    return pc[:, idx, :]


def _chamfer_loss(pc_pred, pc_target, num_points: Optional[int]):
    pc_pred = _sample_points(pc_pred, num_points)
    pc_target = _sample_points(pc_target, num_points)
    dist = ((pc_pred.unsqueeze(2) - pc_target.unsqueeze(1)) ** 2.0).sum(dim=-1)
    pred_to_target, _ = jt.topk(dist, k=1, dim=2, largest=False)
    target_to_pred, _ = jt.topk(dist, k=1, dim=1, largest=False)
    return pred_to_target.mean() + target_to_pred.mean()


def _density_matching_loss(pc_pred, pc_target, k: int, num_points: Optional[int]):
    if k <= 0:
        return 0.0
    pc_pred = _sample_points(pc_pred, num_points)
    pc_target = _sample_points(pc_target, num_points)
    n_pred = pc_pred.shape[1]
    n_target = pc_target.shape[1]
    k_pred = min(k + 1, n_pred)
    k_target = min(k + 1, n_target)
    if k_pred <= 1 or k_target <= 1:
        return 0.0

    pred_dist = ((pc_pred.unsqueeze(2) - pc_pred.unsqueeze(1)) ** 2.0).sum(dim=-1)
    target_dist = ((pc_target.unsqueeze(2) - pc_target.unsqueeze(1)) ** 2.0).sum(dim=-1)
    pred_knn, _ = jt.topk(pred_dist, k=k_pred, dim=-1, largest=False)
    target_knn, _ = jt.topk(target_dist, k=k_target, dim=-1, largest=False)
    pred_radius = pred_knn[:, :, 1:].mean(dim=-1)
    target_radius = target_knn[:, :, 1:].mean(dim=-1)
    pred_radius_sorted, _ = jt.topk(pred_radius, k=pred_radius.shape[1], dim=-1, largest=False)
    target_radius_sorted, _ = jt.topk(target_radius, k=target_radius.shape[1], dim=-1, largest=False)
    m = min(pred_radius_sorted.shape[1], target_radius_sorted.shape[1])
    return ((pred_radius_sorted[:, :m] - target_radius_sorted[:, :m]) ** 2.0).mean()


class CDRefineModule(ModelSpec):
    """
    Second-stage CD-oriented refinement on top of frozen CVM+DM outputs.

    Training reads pc_stage1 from a refine cache and predicts a small residual:
        pc_final = pc_stage1 + delta

    For CD-oriented training, prefer pc_clean_corr as the target_field. It is
    the original clean counterpart of each noisy point, while pc_clean can stay
    as the nearest-surface anchor for P2S preservation.

    Prediction can wrap the full noisy -> CVM+DM -> CDRefine pipeline when a
    stage1 checkpoint is provided in the model config.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.frame_knn = cfg.get("frame_knn", 16)
        self.feat_embedding_dim = cfg.get("feat_embedding_dim", 128)
        self.decoder_hidden_dim = cfg.get("decoder_hidden_dim", 64)
        self.delta_scale = cfg.get("delta_scale", 0.02)
        self.dsm_sigma = cfg.get("dsm_sigma", 0.01)
        self.chamfer_num_points = cfg.get("chamfer_num_points", 256)
        self.chamfer_loss_weight = cfg.get("chamfer_loss_weight", 1.0)
        self.residual_anchor_weight = cfg.get("residual_anchor_weight", 0.02)
        self.point_anchor_weight = cfg.get("point_anchor_weight", 0.05)
        self.surface_anchor_weight = cfg.get("surface_anchor_weight", 0.05)
        self.normal_delta_weight = cfg.get("normal_delta_weight", 0.0)
        self.density_loss_weight = cfg.get("density_loss_weight", 0.0)
        self.density_k = cfg.get("density_k", 8)
        self.density_num_points = cfg.get("density_num_points", self.chamfer_num_points)
        self.condition_mode = cfg.get("condition_mode", "stage1")
        self.condition_scale = cfg.get("condition_scale", 1.0)
        self.allow_direct_refine = cfg.get("allow_direct_refine", False)
        self.target_field = cfg.get("target_field", "pc_clean_corr")
        self.fallback_target_field = cfg.get("fallback_target_field", "pc_clean")
        self.surface_field = cfg.get("surface_field", "pc_clean")
        self.normal_field = cfg.get("normal_field", "pc_normal")
        self.normal_source_field = cfg.get("normal_source_field", "pc_noisy")

        if self.condition_mode == "stage1":
            input_dim = 3
        elif self.condition_mode == "noisy_delta":
            input_dim = 6
        elif self.condition_mode == "noisy_full":
            input_dim = 9
        else:
            raise ValueError(f"unsupported CDRefine condition_mode: {self.condition_mode}")

        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=input_dim,
            embedding_dim=self.feat_embedding_dim,
            distance_estimation=cfg.get("normalize_features", True),
        )
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=self.decoder_hidden_dim,
        )

        self.stage1_ckpt = cfg.get("stage1_ckpt", cfg.get("cvm_dm_ckpt", None))
        self.stage1_model = None
        if self.stage1_ckpt is not None:
            stage1_cfg = cfg.get("stage1_model", None)
            if stage1_cfg is None:
                stage1_cfg = _default_stage1_config()
            self.stage1_model = StraightPCFVelocityDistanceModule(
                model_config=_without_target(stage1_cfg),
                transform_config=transform_config,
            )
            self.stage1_model.load(self.stage1_ckpt)
            self._freeze_stage1_model()

    def _freeze_stage1_model(self):
        self.stage1_model.eval()
        for param in self.stage1_model.parameters():
            if hasattr(param, "stop_grad"):
                param.stop_grad()
            if hasattr(param, "requires_grad"):
                param.requires_grad = False

    def _refine_input(self, pc_stage1, pc_noisy=None):
        if self.condition_mode == "stage1":
            return pc_stage1
        if pc_noisy is None:
            raise ValueError(f"condition_mode={self.condition_mode} requires pc_noisy")
        noisy_delta = self.condition_scale * (pc_noisy - pc_stage1)
        if self.condition_mode == "noisy_delta":
            return jt.concat([pc_stage1, noisy_delta], dim=-1)
        if self.condition_mode == "noisy_full":
            return jt.concat([pc_stage1, pc_noisy, noisy_delta], dim=-1)
        raise ValueError(f"unsupported CDRefine condition_mode: {self.condition_mode}")

    def _predict_delta(self, pc_stage1, pc_noisy=None):
        B, N, d = pc_stage1.shape
        refine_input = self._refine_input(pc_stage1, pc_noisy=pc_noisy)
        feat = self.encoder(refine_input)
        F_dim = feat.shape[-1]
        raw_delta = self.decoder(
            c=feat.reshape(-1, F_dim),
        ).reshape(B, N, d)
        return self.delta_scale * jt.tanh(raw_delta)

    def refine(self, pc_stage1, pc_noisy=None):
        delta = self._predict_delta(pc_stage1, pc_noisy=pc_noisy)
        return pc_stage1 + delta, delta

    def get_supervised_loss(self, pc_stage1, pc_target, pc_surface=None, pc_normal_proxy=None, pc_noisy=None):
        if pc_surface is None:
            pc_surface = pc_target
        pc_final, delta = self.refine(pc_stage1, pc_noisy=pc_noisy)
        chamfer = _chamfer_loss(
            pc_pred=pc_final,
            pc_target=pc_target,
            num_points=self.chamfer_num_points,
        )
        residual_anchor = (delta ** 2.0).sum(dim=-1).mean()
        point_anchor = ((pc_final - pc_target) ** 2.0).sum(dim=-1).mean()
        surface_anchor = ((pc_final - pc_surface) ** 2.0).sum(dim=-1).mean()
        normal_delta = 0.0
        if pc_normal_proxy is not None and self.normal_delta_weight > 0:
            norm = jt.sqrt((pc_normal_proxy ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
            normal = pc_normal_proxy / norm
            normal_delta = ((delta * normal).sum(dim=-1) ** 2.0).mean()
        density_loss = 0.0
        if self.density_loss_weight > 0:
            density_loss = _density_matching_loss(
                pc_pred=pc_final,
                pc_target=pc_target,
                k=self.density_k,
                num_points=self.density_num_points,
            )
        return (
            self.chamfer_loss_weight * chamfer +
            self.residual_anchor_weight * residual_anchor +
            self.point_anchor_weight * point_anchor +
            self.surface_anchor_weight * surface_anchor +
            self.normal_delta_weight * normal_delta +
            self.density_loss_weight * density_loss
        ) / self.dsm_sigma

    def _run_stage1(self, pcl_noisy, num_steps: int=None):
        if self.stage1_model is None:
            if self.allow_direct_refine:
                return pcl_noisy
            raise RuntimeError(
                "CDRefineModule prediction requires stage1_ckpt/stage1_model. "
                "Set allow_direct_refine=True only for direct-refine ablations."
            )
        self.stage1_model.eval()
        with jt.no_grad():
            pc_stage1, _ = self.stage1_model.denoise_langevin_dynamics(
                pcl_noisy,
                num_steps=num_steps,
            )
        return pc_stage1

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None):
        with jt.no_grad():
            pc_stage1 = self._run_stage1(pcl_noisy, num_steps=num_steps)
            pc_final, delta = self.refine(pc_stage1, pc_noisy=pcl_noisy)
        return pc_final, delta

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_stage1"].shape[-2]
        pc_stage1 = batch["pc_stage1"].reshape(-1, patch_size, 3)
        pc_target = batch["pc_refine_target"].reshape(-1, patch_size, 3)
        pc_surface = batch.get("pc_surface", None)
        if pc_surface is not None:
            pc_surface = pc_surface.reshape(-1, patch_size, 3)
        pc_normal_proxy = batch.get("pc_normal_proxy", None)
        if pc_normal_proxy is not None:
            pc_normal_proxy = pc_normal_proxy.reshape(-1, patch_size, 3)
        pc_noisy = batch.get("pc_noisy", None)
        if pc_noisy is not None:
            pc_noisy = pc_noisy.reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_stage1=pc_stage1,
            pc_target=pc_target,
            pc_surface=pc_surface,
            pc_normal_proxy=pc_normal_proxy,
            pc_noisy=pc_noisy,
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
                if "pc_stage1" not in b.meta:
                    raise KeyError(
                        f"{b.path} does not contain pc_stage1. "
                        "Build a refine cache with tools/build_refine_cache.py first."
                    )
                target_key = self.target_field
                if target_key not in b.meta:
                    target_key = self.fallback_target_field
                if target_key not in b.meta:
                    raise KeyError(
                        f"{b.path} does not contain {self.target_field} or "
                        f"{self.fallback_target_field} for CDRefine supervision."
                    )
                d = {
                    "pc_stage1": b.meta["pc_stage1"],
                    "pc_refine_target": b.meta[target_key],
                }
                if self.surface_field in b.meta:
                    d["pc_surface"] = b.meta[self.surface_field]
                elif "pc_clean" in b.meta:
                    d["pc_surface"] = b.meta["pc_clean"]
                if self.normal_field in b.meta:
                    d["pc_normal_proxy"] = b.meta[self.normal_field]
                elif (
                    self.normal_source_field in b.meta and
                    "pc_surface" in d
                ):
                    d["pc_normal_proxy"] = b.meta[self.normal_source_field] - d["pc_surface"]
                for optional_key in ("pc_noisy", "pc_mix", "pc_time"):
                    if optional_key in b.meta:
                        d[optional_key] = b.meta[optional_key]
                res.append(d)
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy,
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res


class TangentialCDRefineModule(CDRefineModule):
    pass
