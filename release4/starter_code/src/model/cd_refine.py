from copy import deepcopy
from typing import Dict, List, Optional

import jittor as jt
import numpy as np
from jittor import nn

from .feature import Decoder, FeatureExtraction, get_knn_idx
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


def _one_sided_nn_loss(pc_pred, pc_target, num_points: Optional[int]):
    pc_pred = _sample_points(pc_pred, num_points)
    pc_target = _sample_points(pc_target, num_points)
    dist = ((pc_pred.unsqueeze(2) - pc_target.unsqueeze(1)) ** 2.0).sum(dim=-1)
    pred_to_target, _ = jt.topk(dist, k=1, dim=2, largest=False)
    return pred_to_target.mean()


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


class LocalFeatureAttention(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        k: int,
        hidden_dim: int,
        residual_scale: float=1.0,
        use_edge_risk: bool=True,
    ):
        super().__init__()
        self.k = k
        self.feat_dim = feat_dim
        self.residual_scale = residual_scale
        self.use_edge_risk = use_edge_risk
        score_dim = 2 * feat_dim + 7
        value_dim = feat_dim + 4
        self.score_mlp = nn.Sequential(
            nn.Linear(score_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(value_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(2 * feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def _edge_index(self, pc, k: int):
        B, N, _ = pc.shape
        knn_idx = get_knn_idx(pc, pc, k, offset=1)
        base = jt.arange(B) * N
        base = base.reshape(B, 1, 1)
        knn_idx = knn_idx + base

        dst = jt.arange(N)
        dst = dst.reshape(1, N, 1).broadcast((B, N, k))
        dst = dst + base
        return knn_idx.reshape(-1), dst.reshape(-1)

    def _radius_risk(self, radius):
        mean = radius.mean(dim=1, keepdims=True)
        var = ((radius - mean) ** 2.0).mean(dim=1, keepdims=True)
        return jt.sigmoid((radius - mean) / jt.sqrt(var + 1e-8))

    def execute(self, pc, feat, pc_edge_risk=None):
        B, N, _ = pc.shape
        if N <= 1 or self.k <= 0:
            return feat

        k = min(self.k, N - 1)
        src, dst = self._edge_index(pc, k)
        feat_flat = feat.reshape(B * N, self.feat_dim)
        pc_flat = pc.reshape(B * N, 3)

        feat_i = feat_flat[dst]
        feat_j = feat_flat[src]
        pos_delta = pc_flat[src] - pc_flat[dst]
        dist2 = (pos_delta ** 2.0).sum(dim=1, keepdims=True)

        radius_flat = jt.full((B * N, 1), 0.0)
        radius_flat = radius_flat.scatter_(
            0,
            dst.unsqueeze(1).broadcast(dist2.shape),
            dist2,
            reduce='add',
        )
        radius = (radius_flat / float(k)).reshape(B, N, 1)
        risk = self._radius_risk(radius)
        if self.use_edge_risk and pc_edge_risk is not None:
            risk = pc_edge_risk
        risk_flat = risk.reshape(B * N, 1)
        risk_i = risk_flat[dst]
        risk_j = risk_flat[src]
        risk_diff2 = (risk_i - risk_j) ** 2.0

        score_input = jt.concat(
            [feat_i, feat_j, pos_delta, dist2, risk_i, risk_j, risk_diff2],
            dim=1,
        )
        score = jt.sigmoid(self.score_mlp(score_input))

        value_input = jt.concat([feat_j, pos_delta, risk_j], dim=1)
        value = self.value_mlp(value_input)
        weighted = score * value

        agg = jt.full((B * N, self.feat_dim), 0.0)
        denom = jt.full((B * N, 1), 0.0)
        agg = agg.scatter_(
            0,
            dst.unsqueeze(1).broadcast(weighted.shape),
            weighted,
            reduce='add',
        )
        denom = denom.scatter_(
            0,
            dst.unsqueeze(1).broadcast(score.shape),
            score,
            reduce='add',
        )
        agg = agg / (denom + 1e-6)
        agg = agg.reshape(B, N, self.feat_dim)
        refined = self.out_proj(jt.concat([feat, agg], dim=-1))
        return feat + self.residual_scale * refined


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
        self.surface_set_weight = cfg.get("surface_set_weight", 0.0)
        self.surface_set_num_points = cfg.get("surface_set_num_points", self.chamfer_num_points)
        self.normal_delta_weight = cfg.get("normal_delta_weight", 0.0)
        self.tangent_delta_weight = cfg.get("tangent_delta_weight", 0.0)
        self.edge_point_anchor_weight = cfg.get("edge_point_anchor_weight", 0.0)
        self.edge_residual_anchor_weight = cfg.get("edge_residual_anchor_weight", 0.0)
        self.edge_normal_delta_weight = cfg.get("edge_normal_delta_weight", 0.0)
        self.edge_tangent_delta_weight = cfg.get("edge_tangent_delta_weight", 0.0)
        self.density_loss_weight = cfg.get("density_loss_weight", 0.0)
        self.density_k = cfg.get("density_k", 8)
        self.density_num_points = cfg.get("density_num_points", self.chamfer_num_points)
        self.allow_direct_refine = cfg.get("allow_direct_refine", False)
        self.target_field = cfg.get("target_field", "pc_clean_corr")
        self.fallback_target_field = cfg.get("fallback_target_field", "pc_clean")
        self.surface_field = cfg.get("surface_field", "pc_clean")
        self.normal_field = cfg.get("normal_field", "pc_normal")
        self.normal_source_field = cfg.get("normal_source_field", "pc_noisy")
        self.use_local_attention = cfg.get("use_local_attention", False)
        self.predict_patch_size = cfg.get("predict_patch_size", 1000)
        self.predict_patch_seed_k = cfg.get("predict_patch_seed_k", 6)
        self.predict_patch_seed_k_alpha = cfg.get("predict_patch_seed_k_alpha", 1)
        self.predict_patch_aggregation = cfg.get("predict_patch_aggregation", "best")
        self.predict_patch_weight_temperature = cfg.get("predict_patch_weight_temperature", 1.0)

        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=self.feat_embedding_dim,
            distance_estimation=cfg.get("normalize_features", True),
        )
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=self.decoder_hidden_dim,
        )
        self.local_attention = None
        if self.use_local_attention:
            self.local_attention = LocalFeatureAttention(
                feat_dim=self.encoder.embedding_dim,
                k=cfg.get("attention_k", self.frame_knn),
                hidden_dim=cfg.get("attention_hidden_dim", self.decoder_hidden_dim),
                residual_scale=cfg.get("attention_residual_scale", 1.0),
                use_edge_risk=cfg.get("attention_use_edge_risk", True),
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

    def _predict_delta(self, pc_stage1, pc_edge_risk=None):
        B, N, d = pc_stage1.shape
        feat = self.encoder(pc_stage1)
        if self.local_attention is not None:
            feat = self.local_attention(pc_stage1, feat, pc_edge_risk=pc_edge_risk)
        F_dim = feat.shape[-1]
        raw_delta = self.decoder(
            c=feat.reshape(-1, F_dim),
        ).reshape(B, N, d)
        return self.delta_scale * jt.tanh(raw_delta)

    def refine(self, pc_stage1, pc_edge_risk=None):
        delta = self._predict_delta(pc_stage1, pc_edge_risk=pc_edge_risk)
        return pc_stage1 + delta, delta

    def get_supervised_loss(self, pc_stage1, pc_target, pc_surface=None, pc_normal_proxy=None, pc_edge_risk=None):
        if pc_surface is None:
            pc_surface = pc_target
        pc_final, delta = self.refine(pc_stage1, pc_edge_risk=pc_edge_risk)
        chamfer = _chamfer_loss(
            pc_pred=pc_final,
            pc_target=pc_target,
            num_points=self.chamfer_num_points,
        )
        residual_anchor = (delta ** 2.0).sum(dim=-1).mean()
        point_anchor = ((pc_final - pc_target) ** 2.0).sum(dim=-1).mean()
        surface_anchor = ((pc_final - pc_surface) ** 2.0).sum(dim=-1).mean()
        surface_set = 0.0
        if self.surface_set_weight > 0:
            surface_set = _one_sided_nn_loss(
                pc_pred=pc_final,
                pc_target=pc_surface,
                num_points=self.surface_set_num_points,
            )

        normal = None
        normal_dot = None
        tangent_delta_sq = None
        needs_normal = (
            pc_normal_proxy is not None and
            (
                self.normal_delta_weight > 0 or
                self.tangent_delta_weight > 0 or
                self.edge_normal_delta_weight > 0 or
                self.edge_tangent_delta_weight > 0
            )
        )
        if needs_normal:
            norm = jt.sqrt((pc_normal_proxy ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
            normal = pc_normal_proxy / norm
            normal_dot = (delta * normal).sum(dim=-1, keepdims=True)
            tangent_delta = delta - normal_dot * normal
            tangent_delta_sq = (tangent_delta ** 2.0).sum(dim=-1)

        normal_delta = 0.0
        if normal_dot is not None and self.normal_delta_weight > 0:
            normal_delta = (normal_dot.squeeze(-1) ** 2.0).mean()
        tangent_delta_loss = 0.0
        if tangent_delta_sq is not None and self.tangent_delta_weight > 0:
            tangent_delta_loss = tangent_delta_sq.mean()

        edge_point_anchor = 0.0
        edge_residual_anchor = 0.0
        edge_normal_delta = 0.0
        edge_tangent_delta = 0.0
        if pc_edge_risk is not None:
            edge_risk = pc_edge_risk.squeeze(-1)
            if self.edge_point_anchor_weight > 0:
                edge_point_anchor = (
                    edge_risk * ((pc_final - pc_target) ** 2.0).sum(dim=-1)
                ).mean()
            if self.edge_residual_anchor_weight > 0:
                edge_residual_anchor = (edge_risk * (delta ** 2.0).sum(dim=-1)).mean()
            if (
                normal_dot is not None and
                self.edge_normal_delta_weight > 0
            ):
                edge_normal_delta = (edge_risk * (normal_dot.squeeze(-1) ** 2.0)).mean()
            if tangent_delta_sq is not None and self.edge_tangent_delta_weight > 0:
                edge_tangent_delta = (edge_risk * tangent_delta_sq).mean()
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
            self.surface_set_weight * surface_set +
            self.normal_delta_weight * normal_delta +
            self.tangent_delta_weight * tangent_delta_loss +
            self.edge_point_anchor_weight * edge_point_anchor +
            self.edge_residual_anchor_weight * edge_residual_anchor +
            self.edge_normal_delta_weight * edge_normal_delta +
            self.edge_tangent_delta_weight * edge_tangent_delta +
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
            pc_final, delta = self.refine(pc_stage1)
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
        pc_edge_risk = batch.get("pc_edge_risk", None)
        if pc_edge_risk is not None:
            pc_edge_risk = pc_edge_risk.reshape(-1, patch_size, 1)
        loss = self.get_supervised_loss(
            pc_stage1=pc_stage1,
            pc_target=pc_target,
            pc_surface=pc_surface,
            pc_normal_proxy=pc_normal_proxy,
            pc_edge_risk=pc_edge_risk,
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
                patch_size=self.predict_patch_size,
                seed_k=self.predict_patch_seed_k,
                seed_k_alpha=self.predict_patch_seed_k_alpha,
                aggregation=self.predict_patch_aggregation,
                weight_temperature=self.predict_patch_weight_temperature,
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
                if "pc_edge_risk" in b.meta:
                    d["pc_edge_risk"] = b.meta["pc_edge_risk"]
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


class LocalAttentionCDRefineModule(CDRefineModule):
    def __init__(self, model_config, transform_config):
        cfg = deepcopy(model_config)
        cfg.setdefault("use_local_attention", True)
        cfg.setdefault("attention_use_edge_risk", False)
        super().__init__(cfg, transform_config)


class RiskAwareCDRefineModule(CDRefineModule):
    def __init__(self, model_config, transform_config):
        cfg = deepcopy(model_config)
        cfg.setdefault("use_local_attention", True)
        cfg.setdefault("attention_use_edge_risk", True)
        super().__init__(cfg, transform_config)
