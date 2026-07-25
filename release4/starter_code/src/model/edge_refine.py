from copy import deepcopy
from typing import Dict, List, Optional

import jittor as jt
from jittor import nn

from .cd_refine import (
    CDRefineModule,
    LocalAttentionCDRefineModule,
    LocalFeatureAttention,
    RiskAwareCDRefineModule,
    TangentialCDRefineModule,
    _chamfer_loss,
    _clamp01,
    _default_stage1_config,
    _density_matching_loss,
    _knn_neighbors,
    _normalize_vectors,
    _one_sided_nn_loss,
    _sample_points,
    _sample_points_pair_with_extra,
)
from .feature import FeatureExtraction
from .spec import ModelSpec
from .vm import patch_based_denoise

from ..data.asset import Asset


def _build_cd_model(model_config, transform_config):
    cfg = deepcopy(model_config)
    target = cfg.pop("__target__", "RiskAwareCDRefineModule")
    cls_map = {
        "CDRefineModule": CDRefineModule,
        "LocalAttentionCDRefineModule": LocalAttentionCDRefineModule,
        "RiskAwareCDRefineModule": RiskAwareCDRefineModule,
        "TangentialCDRefineModule": TangentialCDRefineModule,
    }
    if target not in cls_map:
        raise ValueError(f"EdgeRefine base_cd_model expects CDRefine target, found {target}")
    return cls_map[target](model_config=cfg, transform_config=transform_config)


def _squared_norm(x):
    return (x ** 2.0).sum(dim=-1, keepdims=True)


def _weighted_one_sided_loss(pc_pred, pc_target, weights, num_points: Optional[int], weight_power: float=1.0):
    pc_pred, pc_target, weights = _sample_points_pair_with_extra(
        pc_pred,
        pc_target,
        weights,
        num_points,
    )
    dist = ((pc_pred.unsqueeze(2) - pc_target.unsqueeze(1)) ** 2.0).sum(dim=-1)
    pred_to_target, _ = jt.topk(dist, k=1, dim=2, largest=False)

    B = pc_pred.shape[0]
    weights = _clamp01(weights).reshape(B, -1)
    if weight_power != 1.0:
        weights = weights ** float(weight_power)
    pred_to_target = pred_to_target.reshape(B, -1)
    return (pred_to_target * weights).sum() / (weights.sum() + 1e-6)


def _weighted_coverage_loss(pc_pred, pc_target, weights, num_points: Optional[int], weight_power: float=1.0):
    pc_pred, pc_target, weights = _sample_points_pair_with_extra(
        pc_pred,
        pc_target,
        weights,
        num_points,
    )
    dist = ((pc_pred.unsqueeze(2) - pc_target.unsqueeze(1)) ** 2.0).sum(dim=-1)
    target_to_pred, _ = jt.topk(dist, k=1, dim=1, largest=False)

    B = pc_target.shape[0]
    weights = _clamp01(weights).reshape(B, -1)
    if weight_power != 1.0:
        weights = weights ** float(weight_power)
    target_to_pred = target_to_pred.reshape(B, -1)
    return (target_to_pred * weights).sum() / (weights.sum() + 1e-6)


def _repulsion_loss(pc_pred, pc_target, k: int, num_points: Optional[int], margin: float):
    if k <= 0 or margin <= 0:
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
    pred_nn = jt.sqrt(pred_knn[:, :, 1:] + 1e-12)
    target_nn = jt.sqrt(target_knn[:, :, 1:] + 1e-12)

    B = pc_pred.shape[0]
    target_spacing = target_nn.reshape(B, -1).mean(dim=1).reshape(B, 1, 1)
    threshold = float(margin) * target_spacing.broadcast(pred_nn.shape)
    penalty = jt.maximum(threshold - pred_nn, jt.zeros_like(pred_nn))
    return (penalty ** 2.0).mean()


class EdgeRefineModule(ModelSpec):
    """
    Structure-aware residual after CDRefine.

    The base CDRefine stage handles common surface sampling recovery. This
    module learns a gated residual that is encouraged to stay near zero on
    ordinary regions and activate on edge/thin-structure regions.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.frame_knn = cfg.get("frame_knn", 16)
        self.feat_embedding_dim = cfg.get("feat_embedding_dim", 128)
        self.hidden_dim = cfg.get("hidden_dim", 64)
        self.delta_scale = cfg.get("delta_scale", 0.05)
        self.gate_bias = cfg.get("gate_bias", -2.0)
        self.dsm_sigma = cfg.get("dsm_sigma", 0.01)

        self.chamfer_loss_weight = cfg.get("chamfer_loss_weight", 1.0)
        self.chamfer_num_points = cfg.get("chamfer_num_points", 512)
        self.edge_chamfer_loss_weight = cfg.get("edge_chamfer_loss_weight", 1.0)
        self.edge_chamfer_num_points = cfg.get("edge_chamfer_num_points", self.chamfer_num_points)
        self.edge_weight_power = cfg.get("edge_weight_power", 1.5)
        self.edge_one_sided_weight = cfg.get("edge_one_sided_weight", 0.5)
        self.edge_coverage_weight = cfg.get("edge_coverage_weight", 0.5)

        self.surface_set_weight = cfg.get("surface_set_weight", 0.1)
        self.surface_set_num_points = cfg.get("surface_set_num_points", self.chamfer_num_points)
        self.density_loss_weight = cfg.get("density_loss_weight", 0.05)
        self.density_k = cfg.get("density_k", 8)
        self.density_num_points = cfg.get("density_num_points", self.chamfer_num_points)
        self.repulsion_loss_weight = cfg.get("repulsion_loss_weight", 0.0)
        self.repulsion_k = cfg.get("repulsion_k", 4)
        self.repulsion_num_points = cfg.get("repulsion_num_points", self.chamfer_num_points)
        self.repulsion_margin = cfg.get("repulsion_margin", 0.55)

        self.residual_anchor_weight = cfg.get("residual_anchor_weight", 0.004)
        self.normal_pass_weight = cfg.get("normal_pass_weight", 0.02)
        self.normal_pass_power = cfg.get("normal_pass_power", 1.0)
        self.gate_supervision_weight = cfg.get("gate_supervision_weight", 0.02)

        self.base_field = cfg.get("base_field", "pc_cd")
        self.stage1_field = cfg.get("stage1_field", "pc_stage1")
        self.target_field = cfg.get("target_field", "pc_clean_corr")
        self.fallback_target_field = cfg.get("fallback_target_field", "pc_clean")
        self.surface_field = cfg.get("surface_field", "pc_clean_corr")
        self.normal_field = cfg.get("normal_field", "pc_normal")

        self.predict_patch_size = cfg.get("predict_patch_size", 1000)
        self.predict_patch_seed_k = cfg.get("predict_patch_seed_k", 8)
        self.predict_patch_seed_k_alpha = cfg.get("predict_patch_seed_k_alpha", 1)
        self.predict_patch_aggregation = cfg.get("predict_patch_aggregation", "best")
        self.predict_patch_weight_temperature = cfg.get("predict_patch_weight_temperature", 1.0)
        self.predict_runtime_edge_risk = cfg.get("predict_runtime_edge_risk", True)
        self.runtime_edge_k = cfg.get("runtime_edge_k", self.frame_knn)

        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=self.feat_embedding_dim,
            distance_estimation=cfg.get("normalize_features", True),
        )
        self.local_attention = LocalFeatureAttention(
            feat_dim=self.feat_embedding_dim,
            k=cfg.get("attention_k", self.frame_knn),
            hidden_dim=cfg.get("attention_hidden_dim", self.hidden_dim),
            residual_scale=cfg.get("attention_residual_scale", 1.0),
            use_edge_risk=cfg.get("attention_use_edge_risk", True),
            use_geometry_gate=cfg.get("attention_use_geometry_gate", True),
            geometry_gate_floor=cfg.get("attention_geometry_gate_floor", 0.25),
            geometry_gate_strength=cfg.get("attention_geometry_gate_strength", 2.0),
        )

        context_dim = 13
        head_dim = self.feat_embedding_dim + context_dim
        self.delta_head = nn.Sequential(
            nn.Linear(head_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(head_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.base_cd_ckpt = cfg.get("base_cd_ckpt", None)
        self.base_cd_model = None
        if self.base_cd_ckpt is not None:
            base_cfg = cfg.get("base_cd_model", None)
            if base_cfg is None:
                base_cfg = {
                    "__target__": "RiskAwareCDRefineModule",
                    "stage1_model": _default_stage1_config(),
                }
            self.base_cd_model = _build_cd_model(base_cfg, transform_config)
            self.base_cd_model.load(self.base_cd_ckpt)
            self._freeze_base_cd_model()

    def _freeze_base_cd_model(self):
        self.base_cd_model.eval()
        for param in self.base_cd_model.parameters():
            if hasattr(param, "stop_grad"):
                param.stop_grad()
            if hasattr(param, "requires_grad"):
                param.requires_grad = False

    def _estimate_runtime_geometry(self, pc):
        if not self.predict_runtime_edge_risk or pc.shape[1] <= 2:
            return None, None
        neighbors = _knn_neighbors(pc, self.runtime_edge_k)
        if neighbors is None:
            return None, None
        pc_edge_risk = self._runtime_structure_risk(pc, neighbors)
        pc_normal_proxy = self._normal_from_neighbors(pc, neighbors)
        return pc_edge_risk, pc_normal_proxy

    def _runtime_structure_risk(self, pc, neighbors):
        delta = neighbors - pc.unsqueeze(2)
        dist2 = (delta ** 2.0).sum(dim=-1)
        radius = dist2.mean(dim=-1, keepdims=True)
        mean = radius.mean(dim=1, keepdims=True)
        var = ((radius - mean) ** 2.0).mean(dim=1, keepdims=True)
        return jt.sigmoid((radius - mean) / jt.sqrt(var + 1e-8))

    def _normal_from_neighbors(self, pc, neighbors):
        k = neighbors.shape[2]
        if k < 2:
            return None
        v1 = neighbors[:, :, 0, :] - pc
        v2 = neighbors[:, :, k // 2, :] - pc
        normal = jt.stack(
            [
                v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0],
            ],
            dim=-1,
        )
        return _normalize_vectors(normal)

    def _build_context(self, pc_noisy, pc_stage1, pc_base, pc_edge_risk):
        B, N, _ = pc_base.shape
        zeros3 = jt.full((B, N, 3), 0.0)
        zeros1 = jt.full((B, N, 1), 0.0)

        if pc_noisy is None:
            pc_noisy = pc_base
        if pc_stage1 is None:
            pc_stage1 = pc_base
        if pc_edge_risk is None:
            pc_edge_risk = zeros1

        noisy_to_stage1 = pc_noisy - pc_stage1
        base_to_stage1 = pc_base - pc_stage1
        base_to_noisy = pc_base - pc_noisy
        return jt.concat(
            [
                noisy_to_stage1,
                base_to_stage1,
                base_to_noisy,
                pc_edge_risk,
                jt.sqrt(_squared_norm(noisy_to_stage1) + 1e-12),
                jt.sqrt(_squared_norm(base_to_stage1) + 1e-12),
                jt.sqrt(_squared_norm(base_to_noisy) + 1e-12),
            ],
            dim=-1,
        )

    def refine(self, pc_noisy, pc_stage1, pc_base, pc_edge_risk=None, pc_normal_proxy=None):
        feat = self.encoder(pc_base)
        feat = self.local_attention(
            pc_base,
            feat,
            pc_edge_risk=pc_edge_risk,
            pc_normal_proxy=pc_normal_proxy,
        )
        context = self._build_context(pc_noisy, pc_stage1, pc_base, pc_edge_risk)
        head_input = jt.concat([feat, context], dim=-1)
        B, N, _ = head_input.shape
        head_input = head_input.reshape(B * N, -1)

        delta = float(self.delta_scale) * jt.tanh(self.delta_head(head_input)).reshape(B, N, 3)
        gate = jt.sigmoid(self.gate_head(head_input).reshape(B, N, 1) + float(self.gate_bias))
        correction = gate * delta
        return pc_base + correction, correction, gate

    def get_supervised_loss(
        self,
        pc_noisy,
        pc_stage1,
        pc_base,
        pc_target,
        pc_surface=None,
        pc_edge_risk=None,
        pc_normal_proxy=None,
    ):
        if pc_surface is None:
            pc_surface = pc_target
        if pc_edge_risk is None:
            pc_edge_risk = jt.full((pc_base.shape[0], pc_base.shape[1], 1), 1.0)

        pc_final, correction, gate = self.refine(
            pc_noisy=pc_noisy,
            pc_stage1=pc_stage1,
            pc_base=pc_base,
            pc_edge_risk=pc_edge_risk,
            pc_normal_proxy=pc_normal_proxy,
        )
        edge_risk = _clamp01(pc_edge_risk)
        if self.edge_weight_power != 1.0:
            edge_risk_weighted = edge_risk ** float(self.edge_weight_power)
        else:
            edge_risk_weighted = edge_risk

        chamfer = _chamfer_loss(pc_final, pc_target, self.chamfer_num_points)
        edge_one_sided = _weighted_one_sided_loss(
            pc_final,
            pc_target,
            edge_risk_weighted,
            self.edge_chamfer_num_points,
        )
        edge_coverage = _weighted_coverage_loss(
            pc_final,
            pc_target,
            edge_risk_weighted,
            self.edge_chamfer_num_points,
        )
        surface_set = _one_sided_nn_loss(pc_final, pc_surface, self.surface_set_num_points)
        density = _density_matching_loss(
            pc_final,
            pc_target,
            k=self.density_k,
            num_points=self.density_num_points,
        )
        repulsion = _repulsion_loss(
            pc_final,
            pc_target,
            k=self.repulsion_k,
            num_points=self.repulsion_num_points,
            margin=self.repulsion_margin,
        )

        residual_anchor = _squared_norm(correction).mean()
        normal_weight = (1.0 - edge_risk)
        if self.normal_pass_power != 1.0:
            normal_weight = normal_weight ** float(self.normal_pass_power)
        normal_pass = (normal_weight * _squared_norm(correction)).mean()
        gate_target = _clamp01(pc_edge_risk)
        gate_supervision = ((gate - gate_target) ** 2.0).mean()

        return (
            self.chamfer_loss_weight * chamfer +
            self.edge_chamfer_loss_weight * (edge_one_sided + edge_coverage) +
            self.edge_one_sided_weight * edge_one_sided +
            self.edge_coverage_weight * edge_coverage +
            self.surface_set_weight * surface_set +
            self.density_loss_weight * density +
            self.repulsion_loss_weight * repulsion +
            self.residual_anchor_weight * residual_anchor +
            self.normal_pass_weight * normal_pass +
            self.gate_supervision_weight * gate_supervision
        ) / self.dsm_sigma

    def _run_base_cd(self, pcl_noisy, num_steps: int=None):
        if self.base_cd_model is None:
            raise RuntimeError("EdgeRefine prediction requires base_cd_ckpt/base_cd_model.")
        self.base_cd_model.eval()
        with jt.no_grad():
            pc_stage1 = self.base_cd_model._run_stage1(pcl_noisy, num_steps=num_steps)
            pc_edge_risk, pc_normal_proxy = self.base_cd_model._estimate_runtime_geometry(pc_stage1)
            pc_base, _ = self.base_cd_model.refine(
                pc_stage1,
                pc_edge_risk=pc_edge_risk,
                pc_normal_proxy=pc_normal_proxy,
            )
        return pc_stage1, pc_base

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None):
        with jt.no_grad():
            pc_stage1, pc_base = self._run_base_cd(pcl_noisy, num_steps=num_steps)
            pc_edge_risk, pc_normal_proxy = self._estimate_runtime_geometry(pc_base)
            pc_final, correction, gate = self.refine(
                pc_noisy=pcl_noisy,
                pc_stage1=pc_stage1,
                pc_base=pc_base,
                pc_edge_risk=pc_edge_risk,
                pc_normal_proxy=pc_normal_proxy,
            )
        return pc_final, correction

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch[self.base_field].shape[-2]
        pc_base = batch[self.base_field].reshape(-1, patch_size, 3)
        pc_stage1 = batch.get(self.stage1_field, pc_base).reshape(-1, patch_size, 3)
        pc_noisy = batch.get("pc_noisy", pc_stage1).reshape(-1, patch_size, 3)
        pc_target = batch["pc_refine_target"].reshape(-1, patch_size, 3)
        pc_surface = batch.get("pc_surface", None)
        if pc_surface is not None:
            pc_surface = pc_surface.reshape(-1, patch_size, 3)
        pc_edge_risk = batch.get("pc_edge_risk", None)
        if pc_edge_risk is not None:
            pc_edge_risk = pc_edge_risk.reshape(-1, patch_size, 1)
        pc_normal_proxy = batch.get("pc_normal_proxy", None)
        if pc_normal_proxy is not None:
            pc_normal_proxy = pc_normal_proxy.reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_stage1=pc_stage1,
            pc_base=pc_base,
            pc_target=pc_target,
            pc_surface=pc_surface,
            pc_edge_risk=pc_edge_risk,
            pc_normal_proxy=pc_normal_proxy,
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
                if self.base_field not in b.meta:
                    raise KeyError(
                        f"{b.path} does not contain {self.base_field}. "
                        "Build an edge-refine cache with tools/build_edge_refine_cache.py first."
                    )
                target_key = self.target_field
                if target_key not in b.meta:
                    target_key = self.fallback_target_field
                if target_key not in b.meta:
                    raise KeyError(
                        f"{b.path} does not contain {self.target_field} or "
                        f"{self.fallback_target_field} for EdgeRefine supervision."
                    )
                d = {
                    self.base_field: b.meta[self.base_field],
                    "pc_refine_target": b.meta[target_key],
                }
                for key in ("pc_noisy", self.stage1_field, "pc_edge_risk"):
                    if key in b.meta:
                        d[key] = b.meta[key]
                if self.surface_field in b.meta:
                    d["pc_surface"] = b.meta[self.surface_field]
                elif "pc_clean" in b.meta:
                    d["pc_surface"] = b.meta["pc_clean"]
                if self.normal_field in b.meta:
                    d["pc_normal_proxy"] = b.meta[self.normal_field]
                for optional_key in ("pc_clean", "pc_clean_corr", "pc_mix", "pc_time"):
                    if optional_key in b.meta:
                        d[optional_key] = b.meta[optional_key]
                res.append(d)
            else:
                d = {"pc_noisy": b.sampled_vertices_noisy}
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res
