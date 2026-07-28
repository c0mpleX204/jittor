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


def _sample_points_pair(pc_a, pc_b, num_points: Optional[int]):
    idx = _random_indices(pc_a.shape[1], num_points)
    if idx is None:
        return pc_a, pc_b
    return pc_a[:, idx, :], pc_b[:, idx, :]


def _sample_points_pair_with_extra(pc_a, pc_b, extra, num_points: Optional[int]):
    idx = _random_indices(pc_a.shape[1], num_points)
    if idx is None:
        return pc_a, pc_b, extra
    return pc_a[:, idx, :], pc_b[:, idx, :], extra[:, idx, :]


def _clamp01(x):
    return jt.minimum(jt.maximum(x, jt.ones_like(x) * 0.0), jt.ones_like(x))


def _normalize_vectors(x):
    return x / jt.sqrt((x ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)


def _cross(a, b):
    return jt.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        dim=-1,
    )


def _knn_neighbors(pc, k: int):
    B, N, _ = pc.shape
    k = min(k, N - 1)
    if k <= 0:
        return None
    idx = get_knn_idx(pc, pc, k, offset=1)
    base = (jt.arange(B) * N).reshape(B, 1, 1)
    idx_flat = (idx + base).reshape(-1)
    pc_flat = pc.reshape(B * N, 3)
    return pc_flat[idx_flat].reshape(B, N, k, 3)


def _gather_batched(points, idx):
    B, N, C = points.shape
    idx_shape = idx.shape
    base = (jt.arange(B) * N).reshape(B, 1, 1)
    while len(base.shape) < len(idx_shape):
        base = base.unsqueeze(-1)
    idx_flat = (idx + base).reshape(-1)
    points_flat = points.reshape(B * N, C)
    return points_flat[idx_flat].reshape(*idx_shape, C)


def _query_context_neighbors(query, context, k: int):
    B, _, _ = query.shape
    _, N, _ = context.shape
    k = min(int(k), N)
    if k <= 0:
        return None, None, None
    idx = get_knn_idx(query, context, k, offset=0)
    neighbors = _gather_batched(context, idx)
    rel = neighbors - query.unsqueeze(2)
    dist = jt.sqrt((rel ** 2.0).sum(dim=-1) + 1e-12)
    return idx, neighbors, dist


def _radius_risk_from_neighbors(pc, neighbors):
    delta = neighbors - pc.unsqueeze(2)
    dist2 = (delta ** 2.0).sum(dim=-1)
    radius = dist2.mean(dim=-1, keepdims=True)
    mean = radius.mean(dim=1, keepdims=True)
    var = ((radius - mean) ** 2.0).mean(dim=1, keepdims=True)
    return jt.sigmoid((radius - mean) / jt.sqrt(var + 1e-8))


def _cross_normal_from_neighbors(pc, neighbors):
    k = neighbors.shape[2]
    if k < 2:
        return None
    v1 = neighbors[:, :, 0, :] - pc
    v2 = neighbors[:, :, k // 2, :] - pc
    normal = _cross(v1, v2)
    return _normalize_vectors(normal)


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


def _one_sided_nn_dist(pc_pred, pc_target, num_points: Optional[int]):
    pc_pred = _sample_points(pc_pred, num_points)
    pc_target = _sample_points(pc_target, num_points)
    dist = ((pc_pred.unsqueeze(2) - pc_target.unsqueeze(1)) ** 2.0).sum(dim=-1)
    pred_to_target, _ = jt.topk(dist, k=1, dim=2, largest=False)
    return pred_to_target.squeeze(-1)


def _weighted_chamfer_loss(
    pc_pred,
    pc_target,
    weights,
    num_points: Optional[int],
    weight_power: float=1.0,
):
    pc_pred, pc_target, weights = _sample_points_pair_with_extra(
        pc_pred,
        pc_target,
        weights,
        num_points,
    )
    dist = ((pc_pred.unsqueeze(2) - pc_target.unsqueeze(1)) ** 2.0).sum(dim=-1)
    pred_to_target, _ = jt.topk(dist, k=1, dim=2, largest=False)
    target_to_pred, _ = jt.topk(dist, k=1, dim=1, largest=False)

    B = pc_pred.shape[0]
    weights = _clamp01(weights).reshape(B, -1)
    if weight_power != 1.0:
        weights = weights ** float(weight_power)

    pred_to_target = pred_to_target.reshape(B, -1)
    target_to_pred = target_to_pred.reshape(B, -1)
    weight_sum = weights.sum() + 1e-6
    pred_loss = (pred_to_target * weights).sum() / weight_sum
    target_loss = (target_to_pred * weights).sum() / weight_sum
    return pred_loss + target_loss


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


def _sliced_wasserstein_loss(
    pc_pred,
    pc_target,
    num_points: Optional[int],
    num_projections: int,
    power: float=2.0,
):
    if num_projections <= 0:
        return 0.0
    pc_pred = _sample_points(pc_pred, num_points)
    pc_target = _sample_points(pc_target, num_points)
    m = min(pc_pred.shape[1], pc_target.shape[1])
    if m <= 1:
        return 0.0
    pc_pred = pc_pred[:, :m, :]
    pc_target = pc_target[:, :m, :]

    B = pc_pred.shape[0]
    dirs = jt.randn((B, int(num_projections), 3))
    dirs = _normalize_vectors(dirs)
    pred_proj = (pc_pred.unsqueeze(2) * dirs.unsqueeze(1)).sum(dim=-1)
    target_proj = (pc_target.unsqueeze(2) * dirs.unsqueeze(1)).sum(dim=-1)
    pred_sorted, _ = jt.topk(pred_proj, k=m, dim=1, largest=False)
    target_sorted, _ = jt.topk(target_proj, k=m, dim=1, largest=False)
    diff = pred_sorted - target_sorted
    if power == 1.0:
        return jt.abs(diff).mean()
    return (diff ** float(power)).mean()


def _partial_sinkhorn_ot_loss(
    pc_pred,
    pc_target,
    num_points: Optional[int],
    radius: float,
    temperature: float,
    num_iters: int,
    dustbin_mass: float,
):
    if num_iters <= 0 or radius <= 0.0 or temperature <= 0.0:
        return 0.0
    pc_pred, pc_target = _sample_points_pair(pc_pred, pc_target, num_points)
    m = min(pc_pred.shape[1], pc_target.shape[1])
    if m <= 1:
        return 0.0
    pc_pred = pc_pred[:, :m, :]
    pc_target = pc_target[:, :m, :]

    # The dustbin lets unreliable long matches pay a fixed reject cost instead
    # of forcing every point into a one-to-one correspondence.
    B = pc_pred.shape[0]
    cost_raw = ((pc_pred.unsqueeze(2) - pc_target.unsqueeze(1)) ** 2.0).sum(dim=-1)
    reject_cost = float(radius) ** 2.0
    cost = jt.minimum(cost_raw, jt.ones_like(cost_raw) * reject_cost)
    right = jt.ones((B, m, 1)) * reject_cost
    bottom = jt.ones((B, 1, m)) * reject_cost
    corner = jt.zeros((B, 1, 1))
    cost_top = jt.concat([cost, right], dim=2)
    cost_bottom = jt.concat([bottom, corner], dim=2)
    cost_aug = jt.concat([cost_top, cost_bottom], dim=1)

    kernel = jt.exp(-cost_aug / float(temperature)) + 1e-8
    total_mass = 1.0 + max(float(dustbin_mass), 1e-6)
    real_mass = 1.0 / (float(m) * total_mass)
    dust_mass = max(float(dustbin_mass), 1e-6) / total_mass
    a = jt.concat(
        [
            jt.ones((B, m)) * real_mass,
            jt.ones((B, 1)) * dust_mass,
        ],
        dim=1,
    )
    b = jt.concat(
        [
            jt.ones((B, m)) * real_mass,
            jt.ones((B, 1)) * dust_mass,
        ],
        dim=1,
    )
    u = jt.ones((B, m + 1))
    v = jt.ones((B, m + 1))
    for _ in range(int(num_iters)):
        kv = (kernel * v.unsqueeze(1)).sum(dim=2)
        u = a / (kv + 1e-8)
        ktu = (kernel * u.unsqueeze(2)).sum(dim=1)
        v = b / (ktu + 1e-8)

    plan = u.unsqueeze(2) * kernel * v.unsqueeze(1)
    return (plan * cost_aug).sum() / float(B)


def _surface_guard_loss(
    pc_stage1,
    pc_final,
    pc_surface,
    num_points: Optional[int],
    margin: float,
):
    idx = _random_indices(pc_final.shape[1], num_points)
    if idx is not None:
        pc_stage1 = pc_stage1[:, idx, :]
        pc_final = pc_final[:, idx, :]
    pc_surface = _sample_points(pc_surface, num_points)
    final_dist = _one_sided_nn_dist(pc_final, pc_surface, None)
    stage1_dist = _one_sided_nn_dist(pc_stage1, pc_surface, None)
    excess = final_dist - stage1_dist - float(margin)
    return jt.maximum(excess, jt.zeros_like(excess)).mean()


def _surface_bank_guard_loss(
    pc_stage1,
    pc_final,
    pc_surface_bank,
    num_points: Optional[int],
    bank_num_points: Optional[int],
    margin: float,
):
    idx = _random_indices(pc_final.shape[1], num_points)
    if idx is not None:
        pc_stage1 = pc_stage1[:, idx, :]
        pc_final = pc_final[:, idx, :]
    pc_surface_bank = _sample_points(pc_surface_bank, bank_num_points)
    final_dist = _one_sided_nn_dist(pc_final, pc_surface_bank, None)
    stage1_dist = _one_sided_nn_dist(pc_stage1, pc_surface_bank, None)
    excess = final_dist - stage1_dist - float(margin)
    return jt.maximum(excess, jt.zeros_like(excess)).mean()


def _project_to_tangent(vec, normal):
    normal = _normalize_vectors(normal)
    normal_dot = (vec * normal).sum(dim=-1, keepdims=True)
    return vec - normal_dot * normal, normal_dot


def _self_knn_distances(pc, k: int):
    B, N, _ = pc.shape
    k = min(int(k) + 1, N)
    if k <= 1:
        return None
    dist2 = ((pc.unsqueeze(2) - pc.unsqueeze(1)) ** 2.0).sum(dim=-1)
    knn2, _ = jt.topk(dist2, k=k, dim=-1, largest=False)
    return jt.sqrt(knn2[:, :, 1:] + 1e-12)


def _local_spacing(pc, k: int):
    neighbors = _knn_neighbors(pc, k)
    if neighbors is None:
        return None, None, None

    rel = pc.unsqueeze(2) - neighbors
    dist = jt.sqrt((rel ** 2.0).sum(dim=-1) + 1e-12)
    return dist.mean(dim=2), dist, neighbors


def _tangent_spread_target(
    pc,
    normal,
    k: int,
    max_step: float,
    dense_power: float,
    pc_reference=None,
    reference_spacing_scale: float=1.0,
):
    local_spacing, dist, neighbors = _local_spacing(pc, k)
    if local_spacing is None:
        return (
            jt.zeros_like(pc),
            jt.zeros((pc.shape[0], pc.shape[1])),
            jt.ones((pc.shape[0], pc.shape[1])) * 1e-3,
        )

    if pc_reference is not None:
        reference_spacing, _, _ = _local_spacing(pc_reference, k)
        if reference_spacing is None:
            target_spacing = local_spacing.mean(dim=1, keepdims=True).broadcast(local_spacing.shape)
        else:
            target_spacing = reference_spacing * float(reference_spacing_scale)
    else:
        target_spacing = local_spacing.mean(dim=1, keepdims=True).broadcast(local_spacing.shape)

    local_spacing = dist.mean(dim=2)
    dense = jt.maximum(
        (target_spacing - local_spacing) / (target_spacing + 1e-8),
        jt.zeros_like(local_spacing),
    )
    if dense_power != 1.0:
        dense = dense ** float(dense_power)

    bandwidth = dist[:, :, -1:] + 1e-6
    weight = jt.exp(-((dist / bandwidth) ** 2.0))
    rel = pc.unsqueeze(2) - neighbors
    direction = rel / (dist.unsqueeze(-1) + 1e-8)
    force = (weight.unsqueeze(-1) * direction).sum(dim=2)
    force_tan, _ = _project_to_tangent(force, normal)
    target_delta = _normalize_vectors(force_tan) * (float(max_step) * dense).unsqueeze(-1)
    return target_delta, dense, target_spacing


def _weighted_mean(value, weight):
    denom = weight.sum() + 1e-6
    return (value * weight).sum() / denom


def _soft_local_transport_delta_loss(
    pc_stage1,
    delta,
    pc_target,
    num_points: Optional[int],
    radius: float,
    temperature: float,
    max_delta: float,
    coverage_beta: float,
    confidence_power: float,
):
    if radius <= 0.0 or temperature <= 0.0 or max_delta <= 0.0:
        return 0.0

    src_idx = _random_indices(pc_stage1.shape[1], num_points)
    if src_idx is not None:
        pc_stage1 = pc_stage1[:, src_idx, :]
        delta = delta[:, src_idx, :]
    target_idx = _random_indices(pc_target.shape[1], num_points)
    if target_idx is not None:
        pc_target = pc_target[:, target_idx, :]

    dist2 = ((pc_stage1.unsqueeze(2) - pc_target.unsqueeze(1)) ** 2.0).sum(dim=-1)
    radius2 = float(radius) ** 2.0
    local_mask = (dist2 <= radius2).float32()
    local_kernel = jt.exp(-dist2 / float(temperature)) * local_mask

    coverage = local_kernel.sum(dim=1, keepdims=True)
    if coverage_beta != 0.0:
        anti_crowd = (coverage + 1e-4) ** (-float(coverage_beta))
        local_kernel = local_kernel * anti_crowd

    denom = local_kernel.sum(dim=2, keepdims=True)
    target_pos = (local_kernel.unsqueeze(-1) * pc_target.unsqueeze(1)).sum(dim=2) / (denom + 1e-8)
    target_delta = target_pos - pc_stage1
    target_norm = jt.sqrt((target_delta ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
    cap = jt.minimum(
        jt.ones_like(target_norm),
        jt.ones_like(target_norm) * float(max_delta) / (target_norm + 1e-8),
    )
    target_delta = target_delta * cap

    confidence = denom.squeeze(-1)
    confidence = confidence / (confidence.mean(dim=1, keepdims=True) + 1e-8)
    confidence = _clamp01(confidence)
    if confidence_power != 1.0:
        confidence = confidence ** float(confidence_power)

    delta_error = ((delta - target_delta) ** 2.0).sum(dim=-1)
    return (confidence * delta_error).sum() / (confidence.sum() + 1e-6)


class LocalFeatureAttention(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        k: int,
        hidden_dim: int,
        residual_scale: float=1.0,
        use_edge_risk: bool=True,
        use_geometry_gate: bool=False,
        geometry_gate_floor: float=0.25,
        geometry_gate_strength: float=2.0,
    ):
        super().__init__()
        self.k = k
        self.feat_dim = feat_dim
        self.residual_scale = residual_scale
        self.use_edge_risk = use_edge_risk
        self.use_geometry_gate = use_geometry_gate
        self.geometry_gate_floor = geometry_gate_floor
        self.geometry_gate_strength = geometry_gate_strength
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

    def execute(self, pc, feat, pc_edge_risk=None, pc_normal_proxy=None):
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
        if self.use_geometry_gate and pc_normal_proxy is not None:
            normal_flat = _normalize_vectors(pc_normal_proxy).reshape(B * N, 3)
            normal_i = normal_flat[dst]
            normal_j = normal_flat[src]
            normal_align = (normal_i * normal_j).sum(dim=1, keepdims=True) ** 2.0
            normal_mismatch = 1.0 - _clamp01(normal_align)
            gap_i = jt.sqrt(((pos_delta * normal_i).sum(dim=1, keepdims=True) ** 2.0) + 1e-12)
            gap_j = jt.sqrt(((pos_delta * normal_j).sum(dim=1, keepdims=True) ** 2.0) + 1e-12)
            side_gap = _clamp01((gap_i + gap_j) / (jt.sqrt(dist2 + 1e-12) + 1e-6))
            pair_risk = jt.maximum(risk_i, risk_j)
            geometry_penalty = pair_risk * (normal_mismatch + side_gap)
            gate = (
                float(self.geometry_gate_floor) +
                (1.0 - float(self.geometry_gate_floor)) *
                jt.exp(-float(self.geometry_gate_strength) * geometry_penalty)
            )
            score = score * gate

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
        self.edge_chamfer_loss_weight = cfg.get("edge_chamfer_loss_weight", 0.0)
        self.edge_chamfer_num_points = cfg.get("edge_chamfer_num_points", self.chamfer_num_points)
        self.edge_chamfer_risk_power = cfg.get("edge_chamfer_risk_power", 1.0)
        self.edge_delta_scale = cfg.get("edge_delta_scale", None)
        self.edge_delta_risk_power = cfg.get("edge_delta_risk_power", 1.0)
        self.density_loss_weight = cfg.get("density_loss_weight", 0.0)
        self.density_k = cfg.get("density_k", 8)
        self.density_num_points = cfg.get("density_num_points", self.chamfer_num_points)
        self.swd_loss_weight = cfg.get("swd_loss_weight", 0.0)
        self.swd_num_points = cfg.get("swd_num_points", self.chamfer_num_points)
        self.swd_num_projections = cfg.get("swd_num_projections", 32)
        self.swd_power = cfg.get("swd_power", 2.0)
        self.local_ot_loss_weight = cfg.get("local_ot_loss_weight", 0.0)
        self.local_ot_num_points = cfg.get("local_ot_num_points", self.chamfer_num_points)
        self.local_ot_radius = cfg.get("local_ot_radius", 0.025)
        self.local_ot_temperature = cfg.get("local_ot_temperature", 0.0002)
        self.local_ot_iters = cfg.get("local_ot_iters", 12)
        self.local_ot_dustbin_mass = cfg.get("local_ot_dustbin_mass", 1.0)
        self.soft_ot_delta_weight = cfg.get("soft_ot_delta_weight", 0.0)
        self.soft_ot_num_points = cfg.get("soft_ot_num_points", self.chamfer_num_points)
        self.soft_ot_radius = cfg.get("soft_ot_radius", 0.06)
        self.soft_ot_temperature = cfg.get("soft_ot_temperature", 0.0005)
        self.soft_ot_max_delta = cfg.get("soft_ot_max_delta", 0.04)
        self.soft_ot_coverage_beta = cfg.get("soft_ot_coverage_beta", 0.75)
        self.soft_ot_confidence_power = cfg.get("soft_ot_confidence_power", 1.0)
        self.surface_guard_weight = cfg.get("surface_guard_weight", 0.0)
        self.surface_guard_num_points = cfg.get("surface_guard_num_points", self.chamfer_num_points)
        self.surface_guard_margin = cfg.get("surface_guard_margin", 0.0)
        self.surface_bank_guard_weight = cfg.get("surface_bank_guard_weight", 0.0)
        self.surface_bank_guard_num_points = cfg.get("surface_bank_guard_num_points", self.chamfer_num_points)
        self.surface_bank_guard_bank_points = cfg.get("surface_bank_guard_bank_points", 4096)
        self.surface_bank_guard_margin = cfg.get("surface_bank_guard_margin", 0.0)
        self.allow_direct_refine = cfg.get("allow_direct_refine", False)
        self.target_field = cfg.get("target_field", "pc_clean_corr")
        self.fallback_target_field = cfg.get("fallback_target_field", "pc_clean")
        self.surface_field = cfg.get("surface_field", "pc_clean")
        self.surface_bank_field = cfg.get("surface_bank_field", "pc_surface_bank")
        self.normal_field = cfg.get("normal_field", "pc_normal")
        self.normal_source_field = cfg.get("normal_source_field", "pc_noisy")
        self.use_local_attention = cfg.get("use_local_attention", False)
        self.predict_patch_size = cfg.get("predict_patch_size", 1000)
        self.predict_patch_seed_k = cfg.get("predict_patch_seed_k", 6)
        self.predict_patch_seed_k_alpha = cfg.get("predict_patch_seed_k_alpha", 1)
        self.predict_patch_aggregation = cfg.get("predict_patch_aggregation", "best")
        self.predict_patch_weight_temperature = cfg.get("predict_patch_weight_temperature", 1.0)
        self.predict_runtime_edge_risk = cfg.get("predict_runtime_edge_risk", False)
        self.runtime_edge_k = cfg.get("runtime_edge_k", self.frame_knn)

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
                use_geometry_gate=cfg.get("attention_use_geometry_gate", False),
                geometry_gate_floor=cfg.get("attention_geometry_gate_floor", 0.25),
                geometry_gate_strength=cfg.get("attention_geometry_gate_strength", 2.0),
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

    def _estimate_runtime_geometry(self, pc_stage1):
        if not self.predict_runtime_edge_risk:
            return None, None
        if pc_stage1.shape[1] <= 2:
            return None, None
        neighbors = _knn_neighbors(pc_stage1, self.runtime_edge_k)
        if neighbors is None:
            return None, None
        pc_edge_risk = _radius_risk_from_neighbors(pc_stage1, neighbors)
        pc_normal_proxy = _cross_normal_from_neighbors(pc_stage1, neighbors)
        return pc_edge_risk, pc_normal_proxy

    def _predict_delta(self, pc_stage1, pc_edge_risk=None, pc_normal_proxy=None):
        B, N, d = pc_stage1.shape
        feat = self.encoder(pc_stage1)
        if self.local_attention is not None:
            feat = self.local_attention(
                pc_stage1,
                feat,
                pc_edge_risk=pc_edge_risk,
                pc_normal_proxy=pc_normal_proxy,
            )
        F_dim = feat.shape[-1]
        raw_delta = self.decoder(
            c=feat.reshape(-1, F_dim),
        ).reshape(B, N, d)
        delta_unit = jt.tanh(raw_delta)
        if self.edge_delta_scale is not None and pc_edge_risk is not None:
            edge_risk = _clamp01(pc_edge_risk)
            if self.edge_delta_risk_power != 1.0:
                edge_risk = edge_risk ** float(self.edge_delta_risk_power)
            scale = (
                float(self.delta_scale) +
                edge_risk * (float(self.edge_delta_scale) - float(self.delta_scale))
            )
            return scale * delta_unit
        return self.delta_scale * delta_unit

    def refine(self, pc_stage1, pc_edge_risk=None, pc_normal_proxy=None):
        delta = self._predict_delta(
            pc_stage1,
            pc_edge_risk=pc_edge_risk,
            pc_normal_proxy=pc_normal_proxy,
        )
        return pc_stage1 + delta, delta

    def get_supervised_loss(
        self,
        pc_stage1,
        pc_target,
        pc_surface=None,
        pc_surface_bank=None,
        pc_normal_proxy=None,
        pc_edge_risk=None,
    ):
        if pc_surface is None:
            pc_surface = pc_target
        pc_final, delta = self.refine(
            pc_stage1,
            pc_edge_risk=pc_edge_risk,
            pc_normal_proxy=pc_normal_proxy,
        )
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
        edge_chamfer = 0.0
        if self.edge_chamfer_loss_weight > 0 and pc_edge_risk is not None:
            edge_chamfer = _weighted_chamfer_loss(
                pc_pred=pc_final,
                pc_target=pc_target,
                weights=pc_edge_risk,
                num_points=self.edge_chamfer_num_points,
                weight_power=self.edge_chamfer_risk_power,
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
        swd_loss = 0.0
        if self.swd_loss_weight > 0:
            swd_loss = _sliced_wasserstein_loss(
                pc_pred=pc_final,
                pc_target=pc_target,
                num_points=self.swd_num_points,
                num_projections=self.swd_num_projections,
                power=self.swd_power,
            )
        local_ot_loss = 0.0
        if self.local_ot_loss_weight > 0:
            local_ot_loss = _partial_sinkhorn_ot_loss(
                pc_pred=pc_final,
                pc_target=pc_target,
                num_points=self.local_ot_num_points,
                radius=self.local_ot_radius,
                temperature=self.local_ot_temperature,
                num_iters=self.local_ot_iters,
                dustbin_mass=self.local_ot_dustbin_mass,
            )
        soft_ot_delta = 0.0
        if self.soft_ot_delta_weight > 0:
            soft_ot_delta = _soft_local_transport_delta_loss(
                pc_stage1=pc_stage1,
                delta=delta,
                pc_target=pc_target,
                num_points=self.soft_ot_num_points,
                radius=self.soft_ot_radius,
                temperature=self.soft_ot_temperature,
                max_delta=self.soft_ot_max_delta,
                coverage_beta=self.soft_ot_coverage_beta,
                confidence_power=self.soft_ot_confidence_power,
            )
        surface_guard = 0.0
        if self.surface_guard_weight > 0:
            surface_guard = _surface_guard_loss(
                pc_stage1=pc_stage1,
                pc_final=pc_final,
                pc_surface=pc_surface,
                num_points=self.surface_guard_num_points,
                margin=self.surface_guard_margin,
            )
        surface_bank_guard = 0.0
        if self.surface_bank_guard_weight > 0 and pc_surface_bank is not None:
            surface_bank_guard = _surface_bank_guard_loss(
                pc_stage1=pc_stage1,
                pc_final=pc_final,
                pc_surface_bank=pc_surface_bank,
                num_points=self.surface_bank_guard_num_points,
                bank_num_points=self.surface_bank_guard_bank_points,
                margin=self.surface_bank_guard_margin,
            )
        return (
            self.chamfer_loss_weight * chamfer +
            self.residual_anchor_weight * residual_anchor +
            self.point_anchor_weight * point_anchor +
            self.surface_anchor_weight * surface_anchor +
            self.surface_set_weight * surface_set +
            self.edge_chamfer_loss_weight * edge_chamfer +
            self.normal_delta_weight * normal_delta +
            self.tangent_delta_weight * tangent_delta_loss +
            self.edge_point_anchor_weight * edge_point_anchor +
            self.edge_residual_anchor_weight * edge_residual_anchor +
            self.edge_normal_delta_weight * edge_normal_delta +
            self.edge_tangent_delta_weight * edge_tangent_delta +
            self.density_loss_weight * density_loss +
            self.swd_loss_weight * swd_loss +
            self.local_ot_loss_weight * local_ot_loss +
            self.soft_ot_delta_weight * soft_ot_delta +
            self.surface_guard_weight * surface_guard +
            self.surface_bank_guard_weight * surface_bank_guard
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
            pc_edge_risk, pc_normal_proxy = self._estimate_runtime_geometry(pc_stage1)
            pc_final, delta = self.refine(
                pc_stage1,
                pc_edge_risk=pc_edge_risk,
                pc_normal_proxy=pc_normal_proxy,
            )
        return pc_final, delta

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_stage1"].shape[-2]
        pc_stage1 = batch["pc_stage1"].reshape(-1, patch_size, 3)
        pc_target = batch["pc_refine_target"].reshape(-1, patch_size, 3)
        pc_surface = batch.get("pc_surface", None)
        if pc_surface is not None:
            pc_surface = pc_surface.reshape(-1, patch_size, 3)
        pc_surface_bank = batch.get("pc_surface_bank", None)
        if pc_surface_bank is not None:
            surface_bank_size = pc_surface_bank.shape[-2]
            pc_surface_bank = pc_surface_bank.reshape(-1, surface_bank_size, 3)
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
            pc_surface_bank=pc_surface_bank,
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
                if self.surface_bank_field in b.meta:
                    d["pc_surface_bank"] = b.meta[self.surface_bank_field]
                if self.normal_field in b.meta:
                    d["pc_normal_proxy"] = b.meta[self.normal_field]
                elif (
                    self.normal_source_field in b.meta and
                    "pc_surface" in d
                ):
                    d["pc_normal_proxy"] = b.meta[self.normal_source_field] - d["pc_surface"]
                if "pc_edge_risk" in b.meta:
                    d["pc_edge_risk"] = b.meta["pc_edge_risk"]
                for optional_key in ("pc_noisy", "pc_mix", "pc_time", "pc_center"):
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


class TangentSpreadRefineModule(ModelSpec):
    """
    Distribution-only refine stage.

    It does not use clean points as correspondence targets. The stage learns a
    small tangent residual that spreads locally crowded stage-1 points. When a
    noisy reference is available, only points that became more crowded than the
    original noisy distribution are encouraged to move. A surface-bank hinge
    prevents moving farther away from the surface proxy.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.frame_knn = cfg.get("frame_knn", 32)
        self.feat_embedding_dim = cfg.get("feat_embedding_dim", 128)
        self.decoder_hidden_dim = cfg.get("decoder_hidden_dim", 64)
        self.delta_scale = cfg.get("delta_scale", 0.012)
        self.dsm_sigma = cfg.get("dsm_sigma", 0.01)
        self.loss_num_points = cfg.get("loss_num_points", 512)
        self.normal_k = cfg.get("normal_k", self.frame_knn)
        self.spread_k = cfg.get("spread_k", 16)
        self.mid_spread_k = cfg.get("mid_spread_k", 0)
        self.repulsion_k = cfg.get("repulsion_k", 8)
        self.force_delta_scale = cfg.get("force_delta_scale", self.delta_scale)
        self.dense_power = cfg.get("dense_power", 1.0)
        self.reference_field = cfg.get("reference_field", "pc_noisy")
        self.reference_spacing_scale = cfg.get("reference_spacing_scale", 1.0)
        self.use_density_delta_gate = cfg.get("use_density_delta_gate", True)
        self.density_gate_floor = cfg.get("density_gate_floor", 0.0)
        self.density_gate_power = cfg.get("density_gate_power", 1.0)
        self.mid_force_weight = cfg.get("mid_force_weight", 0.0)
        self.mid_delta_boost = cfg.get("mid_delta_boost", 0.0)
        self.repulsion_radius_scale = cfg.get("repulsion_radius_scale", 0.72)
        self.spacing_target_scale = cfg.get("spacing_target_scale", 1.0)
        self.reference_overexpand_weight = cfg.get("reference_overexpand_weight", 0.0)
        self.reference_upper_scale = cfg.get("reference_upper_scale", 1.25)
        self.force_loss_weight = cfg.get("force_loss_weight", 0.5)
        self.repulsion_loss_weight = cfg.get("repulsion_loss_weight", 1.0)
        self.spacing_loss_weight = cfg.get("spacing_loss_weight", 0.25)
        self.normal_delta_weight = cfg.get("normal_delta_weight", 0.5)
        self.anchor_loss_weight = cfg.get("anchor_loss_weight", 0.04)
        self.surface_bank_guard_weight = cfg.get("surface_bank_guard_weight", 0.2)
        self.surface_bank_guard_num_points = cfg.get("surface_bank_guard_num_points", self.loss_num_points)
        self.surface_bank_guard_bank_points = cfg.get("surface_bank_guard_bank_points", 2048)
        self.surface_bank_guard_margin = cfg.get("surface_bank_guard_margin", 0.0005)
        self.project_delta_to_tangent = cfg.get("project_delta_to_tangent", True)
        self.allow_direct_refine = cfg.get("allow_direct_refine", False)
        self.normal_field = cfg.get("normal_field", "pc_normal")
        self.surface_bank_field = cfg.get("surface_bank_field", "pc_surface_bank")
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

    def _normal_proxy(self, pc_stage1, pc_normal_proxy=None):
        if pc_normal_proxy is not None:
            return _normalize_vectors(pc_normal_proxy)
        neighbors = _knn_neighbors(pc_stage1, self.normal_k)
        if neighbors is None:
            return jt.zeros_like(pc_stage1)
        normal = _cross_normal_from_neighbors(pc_stage1, neighbors)
        if normal is None:
            return jt.zeros_like(pc_stage1)
        return _normalize_vectors(normal)

    def _predict_delta(self, pc_stage1):
        B, N, d = pc_stage1.shape
        feat = self.encoder(pc_stage1)
        F_dim = feat.shape[-1]
        raw_delta = self.decoder(c=feat.reshape(-1, F_dim)).reshape(B, N, d)
        return float(self.delta_scale) * jt.tanh(raw_delta)

    def _density_gate(self, pc_stage1, normal, pc_reference=None):
        if not self.use_density_delta_gate or pc_reference is None:
            return None

        _, dense_weight, _ = _tangent_spread_target(
            pc_stage1,
            normal,
            k=self.spread_k,
            max_step=1.0,
            dense_power=self.dense_power,
            pc_reference=pc_reference,
            reference_spacing_scale=self.reference_spacing_scale,
        )
        if self.mid_spread_k and self.mid_spread_k > 0 and self.mid_force_weight > 0:
            _, mid_dense, _ = _tangent_spread_target(
                pc_stage1,
                normal,
                k=self.mid_spread_k,
                max_step=1.0,
                dense_power=self.dense_power,
                pc_reference=pc_reference,
                reference_spacing_scale=self.reference_spacing_scale,
            )
            dense_weight = jt.maximum(
                dense_weight,
                mid_dense * float(self.mid_force_weight),
            )

        gate = _clamp01(dense_weight)
        if self.density_gate_power != 1.0:
            gate = gate ** float(self.density_gate_power)
        if self.density_gate_floor > 0.0:
            gate = (
                float(self.density_gate_floor) +
                (1.0 - float(self.density_gate_floor)) * gate
            )
        return gate.unsqueeze(-1)

    def refine(self, pc_stage1, pc_normal_proxy=None, pc_reference=None):
        normal = self._normal_proxy(pc_stage1, pc_normal_proxy=pc_normal_proxy)
        raw_delta = self._predict_delta(pc_stage1)
        if self.project_delta_to_tangent:
            delta, normal_dot = _project_to_tangent(raw_delta, normal)
        else:
            delta = raw_delta
            normal_dot = (raw_delta * normal).sum(dim=-1, keepdims=True)
        density_gate = self._density_gate(
            pc_stage1,
            normal,
            pc_reference=pc_reference,
        )
        if density_gate is not None:
            delta = delta * density_gate
            normal_dot = normal_dot * density_gate
        return pc_stage1 + delta, delta, raw_delta, normal, normal_dot

    def get_distribution_loss(
        self,
        pc_stage1,
        pc_surface_bank=None,
        pc_normal_proxy=None,
        pc_reference=None,
    ):
        pc_final, delta, raw_delta, normal, normal_dot = self.refine(
            pc_stage1,
            pc_normal_proxy=pc_normal_proxy,
            pc_reference=pc_reference,
        )

        idx = _random_indices(pc_stage1.shape[1], self.loss_num_points)
        if idx is not None:
            pc_stage1_l = pc_stage1[:, idx, :]
            pc_final_l = pc_final[:, idx, :]
            delta_l = delta[:, idx, :]
            normal_l = normal[:, idx, :]
            normal_dot_l = normal_dot[:, idx, :]
            pc_reference_l = pc_reference[:, idx, :] if pc_reference is not None else None
        else:
            pc_stage1_l = pc_stage1
            pc_final_l = pc_final
            delta_l = delta
            normal_l = normal
            normal_dot_l = normal_dot
            pc_reference_l = pc_reference

        target_delta, dense_weight, target_spacing = _tangent_spread_target(
            pc_stage1_l,
            normal_l,
            k=self.spread_k,
            max_step=self.force_delta_scale,
            dense_power=self.dense_power,
            pc_reference=pc_reference_l,
            reference_spacing_scale=self.reference_spacing_scale,
        )
        if self.mid_spread_k and self.mid_spread_k > 0 and self.mid_force_weight > 0:
            mid_delta, mid_dense, _ = _tangent_spread_target(
                pc_stage1_l,
                normal_l,
                k=self.mid_spread_k,
                max_step=self.force_delta_scale,
                dense_power=self.dense_power,
                pc_reference=pc_reference_l,
                reference_spacing_scale=self.reference_spacing_scale,
            )
            dense_weight = jt.maximum(
                dense_weight,
                mid_dense * float(self.mid_force_weight),
            )
            target_delta = target_delta + mid_delta * float(self.mid_force_weight)
            if self.mid_delta_boost > 0.0:
                delta_cap = (
                    float(self.force_delta_scale) *
                    (1.0 + float(self.mid_delta_boost) * _clamp01(mid_dense))
                ).unsqueeze(-1)
                target_norm = jt.sqrt((target_delta ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
                target_delta = target_delta * jt.minimum(
                    jt.ones_like(target_norm),
                    delta_cap / (target_norm + 1e-8),
                )
        force_loss = _weighted_mean(
            ((delta_l - target_delta) ** 2.0).sum(dim=-1),
            dense_weight,
        )

        repulsion_loss = 0.0
        spacing_loss = 0.0
        final_knn = _self_knn_distances(pc_final_l, self.repulsion_k)
        if final_knn is not None:
            min_radius = (
                target_spacing.unsqueeze(-1) *
                float(self.repulsion_radius_scale)
            )
            close = jt.maximum(min_radius - final_knn, jt.zeros_like(final_knn))
            repulsion_loss = (close ** 2.0).mean()

            final_spacing = final_knn.mean(dim=2)
            target_spacing = target_spacing * float(self.spacing_target_scale)
            spacing_shortfall = jt.maximum(
                target_spacing - final_spacing,
                jt.zeros_like(final_spacing),
            )
            spacing_loss = _weighted_mean(spacing_shortfall ** 2.0, dense_weight)

        reference_overexpand = 0.0
        if (
            self.reference_overexpand_weight > 0 and
            pc_reference_l is not None and
            final_knn is not None
        ):
            reference_spacing, _, _ = _local_spacing(pc_reference_l, self.repulsion_k)
            if reference_spacing is not None:
                final_spacing = final_knn.mean(dim=2)
                upper_spacing = reference_spacing * float(self.reference_upper_scale)
                over = jt.maximum(
                    final_spacing - upper_spacing,
                    jt.zeros_like(final_spacing),
                )
                reference_overexpand = _weighted_mean(
                    over ** 2.0,
                    jt.ones_like(dense_weight),
                )

        normal_delta = (normal_dot_l.squeeze(-1) ** 2.0).mean()
        anchor = (delta_l ** 2.0).sum(dim=-1).mean()

        surface_bank_guard = 0.0
        if self.surface_bank_guard_weight > 0 and pc_surface_bank is not None:
            surface_bank_guard = _surface_bank_guard_loss(
                pc_stage1=pc_stage1,
                pc_final=pc_final,
                pc_surface_bank=pc_surface_bank,
                num_points=self.surface_bank_guard_num_points,
                bank_num_points=self.surface_bank_guard_bank_points,
                margin=self.surface_bank_guard_margin,
            )

        loss = (
            self.force_loss_weight * force_loss +
            self.repulsion_loss_weight * repulsion_loss +
            self.spacing_loss_weight * spacing_loss +
            self.reference_overexpand_weight * reference_overexpand +
            self.normal_delta_weight * normal_delta +
            self.anchor_loss_weight * anchor +
            self.surface_bank_guard_weight * surface_bank_guard
        ) / self.dsm_sigma
        return loss

    def _run_stage1(self, pcl_noisy, num_steps: int=None):
        if self.stage1_model is None:
            if self.allow_direct_refine:
                return pcl_noisy
            raise RuntimeError(
                "TangentSpreadRefineModule prediction requires stage1_ckpt/stage1_model. "
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
            pc_final, delta, _, _, _ = self.refine(
                pc_stage1,
                pc_reference=pcl_noisy,
            )
        return pc_final, delta

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_stage1"].shape[-2]
        pc_stage1 = batch["pc_stage1"].reshape(-1, patch_size, 3)
        pc_reference = batch.get("pc_reference", None)
        if pc_reference is not None:
            pc_reference = pc_reference.reshape(-1, patch_size, 3)
        pc_surface_bank = batch.get("pc_surface_bank", None)
        if pc_surface_bank is not None:
            surface_bank_size = pc_surface_bank.shape[-2]
            pc_surface_bank = pc_surface_bank.reshape(-1, surface_bank_size, 3)
        pc_normal_proxy = batch.get("pc_normal_proxy", None)
        if pc_normal_proxy is not None:
            pc_normal_proxy = pc_normal_proxy.reshape(-1, patch_size, 3)
        loss = self.get_distribution_loss(
            pc_stage1=pc_stage1,
            pc_surface_bank=pc_surface_bank,
            pc_normal_proxy=pc_normal_proxy,
            pc_reference=pc_reference,
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
                d = {"pc_stage1": b.meta["pc_stage1"]}
                if self.reference_field in b.meta:
                    d["pc_reference"] = b.meta[self.reference_field]
                if self.surface_bank_field in b.meta:
                    d["pc_surface_bank"] = b.meta[self.surface_bank_field]
                if self.normal_field in b.meta:
                    d["pc_normal_proxy"] = b.meta[self.normal_field]
                res.append(d)
            else:
                d = {"pc_noisy": b.sampled_vertices_noisy}
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res


class ScoreFieldRefineModule(TangentSpreadRefineModule):
    """
    Query-conditioned score-field refinement.

    This follows the useful part of Score-Denoise / Deep-RS: train a vector
    field on query positions around stage-1 points, then refine by several
    small gradient-ascent style steps. No GT or surface bank is required at
    prediction time.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config

        self.score_context_knn = cfg.get("score_context_knn", 24)
        self.score_radius = cfg.get("score_radius", 0.12)
        self.score_hidden_dim = cfg.get("score_hidden_dim", self.decoder_hidden_dim)
        self.score_query_points = cfg.get("score_query_points", self.loss_num_points)
        self.score_target_points = cfg.get("score_target_points", 2048)
        self.score_target_avg_knn = cfg.get("score_target_avg_knn", 4)
        self.score_target_max_step = cfg.get("score_target_max_step", 0.035)
        self.score_target_radius = cfg.get("score_target_radius", 0.0)
        self.score_loss_weight = cfg.get("score_loss_weight", 1.0)

        self.query_jitter_tangent_std = cfg.get("query_jitter_tangent_std", 0.006)
        self.query_jitter_normal_std = cfg.get("query_jitter_normal_std", 0.002)

        self.score_steps = cfg.get("score_steps", 4)
        self.score_step_size = cfg.get("score_step_size", 0.35)
        self.score_step_decay = cfg.get("score_step_decay", 0.72)
        self.score_step_cap = cfg.get("score_step_cap", self.delta_scale)
        self.score_total_cap = cfg.get("score_total_cap", 0.035)
        self.normal_step_scale = cfg.get("normal_step_scale", 0.45)

        self.score_target_field = cfg.get("score_target_field", "pc_surface_bank")
        self.score_fallback_target_field = cfg.get("score_fallback_target_field", "pc_clean_corr")
        self.context_feature_k = cfg.get("context_feature_k", self.spread_k)
        self.context_reference_spacing_scale = cfg.get("context_reference_spacing_scale", 1.0)
        self.distribution_spacing_source = cfg.get("distribution_spacing_source", "reference")
        self.hole_spacing_scale = cfg.get("hole_spacing_scale", 1.15)
        self.hole_power = cfg.get("hole_power", 1.0)
        self.hole_upper_scale = cfg.get("hole_upper_scale", 1.18)

        self.repulsion_loss_weight = cfg.get("repulsion_loss_weight", 0.08)
        self.spacing_loss_weight = cfg.get("spacing_loss_weight", 0.04)
        self.hole_loss_weight = cfg.get("hole_loss_weight", 0.04)
        self.normal_delta_weight = cfg.get("normal_delta_weight", 0.25)
        self.anchor_loss_weight = cfg.get("anchor_loss_weight", 0.02)
        self.surface_bank_guard_weight = cfg.get("surface_bank_guard_weight", 0.0)
        self.surface_bank_guard_margin = cfg.get("surface_bank_guard_margin", 0.00004)

        self.score_extra_dim = 10
        point_in_dim = self.encoder.embedding_dim + self.score_extra_dim + 4
        hidden = self.score_hidden_dim
        self.score_point_lin_1 = nn.Linear(point_in_dim, hidden)
        self.score_point_bn_1 = nn.BatchNorm1d(hidden)
        self.score_point_lin_2 = nn.Linear(hidden, hidden)
        self.score_point_bn_2 = nn.BatchNorm1d(hidden)
        self.score_global_lin_1 = nn.Linear(hidden, hidden)
        self.score_global_bn_1 = nn.BatchNorm1d(hidden)
        self.score_global_lin_2 = nn.Linear(hidden, hidden)
        self.score_global_bn_2 = nn.BatchNorm1d(hidden)
        self.score_global_lin_3 = nn.Linear(hidden, 3)
        self.score_act = nn.ReLU()
        self._freeze_unused_delta_decoder()

    def _freeze_unused_delta_decoder(self):
        for param in self.decoder.parameters():
            if hasattr(param, "stop_grad"):
                param.stop_grad()
            if hasattr(param, "requires_grad"):
                param.requires_grad = False

    def _context_extra(self, pc_stage1, normal, pc_reference=None):
        B, N, _ = pc_stage1.shape
        zero = jt.zeros((B, N, 1))
        zero_vec = jt.zeros_like(pc_stage1)

        stage_spacing, _, _ = _local_spacing(pc_stage1, self.context_feature_k)
        if stage_spacing is None:
            stage_spacing = jt.ones((B, N)) * 1e-3

        if pc_reference is None:
            residual_tan = zero_vec
            residual_normal_abs = zero
            reference_spacing = stage_spacing
        else:
            residual = pc_reference - pc_stage1
            residual_tan, residual_normal = _project_to_tangent(residual, normal)
            residual_normal_abs = jt.abs(residual_normal)
            pc_reference_tangent = pc_stage1 + residual_tan
            reference_spacing, _, _ = _local_spacing(pc_reference_tangent, self.context_feature_k)
            if reference_spacing is None:
                reference_spacing = stage_spacing

        residual_norm = jt.sqrt((residual_tan ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
        stage_spacing_u = stage_spacing.unsqueeze(-1)
        reference_spacing_u = reference_spacing.unsqueeze(-1)
        spacing_ratio = stage_spacing_u / (reference_spacing_u + 1e-8)
        dense = jt.maximum(
            (reference_spacing_u * float(self.context_reference_spacing_scale) - stage_spacing_u) /
            (reference_spacing_u + 1e-8),
            jt.zeros_like(stage_spacing_u),
        )
        sparse = jt.maximum(
            (stage_spacing_u - reference_spacing_u * float(self.hole_spacing_scale)) /
            (reference_spacing_u + 1e-8),
            jt.zeros_like(stage_spacing_u),
        )

        return jt.concat(
            [
                residual_tan,
                residual_norm,
                residual_normal_abs,
                stage_spacing_u,
                reference_spacing_u,
                spacing_ratio,
                dense,
                sparse,
            ],
            dim=-1,
        )

    def _score_weights(self, dist):
        if self.score_radius and self.score_radius > 0:
            ratio = jt.minimum(
                dist / float(self.score_radius),
                jt.ones_like(dist),
            )
            weight = 0.5 * (jt.cos(ratio * np.pi) + 1.0)
            weight = weight * (dist <= float(self.score_radius)).float32()
        else:
            bandwidth = dist[:, :, -1:] + 1e-6
            weight = jt.exp(-((dist / bandwidth) ** 2.0))
        weight = weight * (dist > 1e-8).float32()
        return weight

    def _build_context_state(self, pc_stage1, pc_reference=None, pc_normal_proxy=None):
        normal = self._normal_proxy(pc_stage1, pc_normal_proxy=pc_normal_proxy)
        feat = self.encoder(pc_stage1)
        extra = self._context_extra(
            pc_stage1,
            normal,
            pc_reference=pc_reference,
        )
        return normal, feat, extra

    def _predict_score_from_context(self, query, pc_context, feat, extra):
        B, Q, _ = query.shape
        idx, neighbors, dist = _query_context_neighbors(
            query,
            pc_context,
            self.score_context_knn,
        )
        if idx is None:
            return jt.zeros_like(query)

        feat_group = _gather_batched(feat, idx)
        extra_group = _gather_batched(extra, idx)
        rel = neighbors - query.unsqueeze(2)
        point_input = jt.concat(
            [rel, dist.unsqueeze(-1), feat_group, extra_group],
            dim=-1,
        )

        net = point_input.reshape(B * Q * idx.shape[2], -1)
        net = self.score_point_lin_1(net)
        net = self.score_point_bn_1(net)
        net = self.score_act(net)
        net = self.score_point_lin_2(net)
        net = self.score_point_bn_2(net)
        net = self.score_act(net)
        net = net.reshape(B, Q, idx.shape[2], -1)

        weight = self._score_weights(dist).unsqueeze(-1)
        agg = (net * weight).sum(dim=2) / (weight.sum(dim=2) + 1e-6)
        out = agg.reshape(B * Q, -1)
        out = self.score_global_lin_1(out)
        out = self.score_global_bn_1(out)
        out = self.score_act(out)
        out = self.score_global_lin_2(out)
        out = self.score_global_bn_2(out)
        out = self.score_act(out)
        out = self.score_global_lin_3(out).reshape(B, Q, 3)
        return out

    def predict_score(self, query, pc_stage1, pc_reference=None, pc_normal_proxy=None):
        _, feat, extra = self._build_context_state(
            pc_stage1,
            pc_reference=pc_reference,
            pc_normal_proxy=pc_normal_proxy,
        )
        return self._predict_score_from_context(query, pc_stage1, feat, extra)

    def _sample_queries(self, pc_stage1, normal):
        idx = _random_indices(pc_stage1.shape[1], self.score_query_points)
        if idx is None:
            query = pc_stage1
            query_normal = normal
        else:
            query = pc_stage1[:, idx, :]
            query_normal = normal[:, idx, :]

        if self.query_jitter_tangent_std > 0 or self.query_jitter_normal_std > 0:
            jitter = jt.randn(query.shape)
            jitter_tan, _ = _project_to_tangent(jitter, query_normal)
            query = query + float(self.query_jitter_tangent_std) * jitter_tan
            if self.query_jitter_normal_std > 0:
                normal_jitter = jt.randn((query.shape[0], query.shape[1], 1))
                query = query + float(self.query_jitter_normal_std) * normal_jitter * query_normal
        return query

    def _pointset_score_target(self, query, pc_target):
        pc_target = _sample_points(pc_target, self.score_target_points)
        k = min(int(self.score_target_avg_knn), pc_target.shape[1])
        if k <= 0:
            return jt.zeros_like(query), jt.ones((query.shape[0], query.shape[1]))

        dist2 = ((query.unsqueeze(2) - pc_target.unsqueeze(1)) ** 2.0).sum(dim=-1)
        knn2, idx = jt.topk(dist2, k=k, dim=2, largest=False)
        target_nbs = _gather_batched(pc_target, idx)
        target_score = target_nbs.mean(dim=2) - query

        if self.score_target_max_step and self.score_target_max_step > 0:
            norm = jt.sqrt((target_score ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
            target_score = target_score * jt.minimum(
                jt.ones_like(norm),
                jt.ones_like(norm) * float(self.score_target_max_step) / (norm + 1e-8),
            )

        weight = jt.ones((query.shape[0], query.shape[1]))
        if self.score_target_radius and self.score_target_radius > 0:
            nearest = jt.sqrt(knn2[:, :, 0] + 1e-12)
            weight = jt.exp(-((nearest / float(self.score_target_radius)) ** 2.0))
        return target_score, weight

    def _target_spacing_at_queries(self, query, pc_target, k: int):
        target_spacing, _, _ = _local_spacing(pc_target, k)
        if target_spacing is None:
            return None
        dist2 = ((query.unsqueeze(2) - pc_target.unsqueeze(1)) ** 2.0).sum(dim=-1)
        _, idx = jt.topk(dist2, k=1, dim=2, largest=False)
        spacing = _gather_batched(target_spacing.unsqueeze(-1), idx)
        return spacing.squeeze(2).squeeze(-1)

    def _apply_score_step(self, pc_current, pc_context, feat, extra, normal, step_scale):
        score = self._predict_score_from_context(
            pc_current,
            pc_context,
            feat,
            extra,
        )
        score_tan, normal_dot = _project_to_tangent(score, normal)
        score = score_tan + float(self.normal_step_scale) * normal_dot * normal
        delta_step = float(step_scale) * score
        if self.score_step_cap and self.score_step_cap > 0:
            norm = jt.sqrt((delta_step ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
            delta_step = delta_step * jt.minimum(
                jt.ones_like(norm),
                jt.ones_like(norm) * float(self.score_step_cap) / (norm + 1e-8),
            )
        return delta_step, score

    def _refine_with_context(self, pc_stage1, feat, extra, normal):
        pc_current = pc_stage1
        last_score = jt.zeros_like(pc_stage1)
        step = float(self.score_step_size)
        for _ in range(int(self.score_steps)):
            delta_step, last_score = self._apply_score_step(
                pc_current,
                pc_stage1,
                feat,
                extra,
                normal,
                step,
            )
            pc_current = pc_current + delta_step
            step *= float(self.score_step_decay)

        delta = pc_current - pc_stage1
        if self.score_total_cap and self.score_total_cap > 0:
            norm = jt.sqrt((delta ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
            delta = delta * jt.minimum(
                jt.ones_like(norm),
                jt.ones_like(norm) * float(self.score_total_cap) / (norm + 1e-8),
            )
            pc_current = pc_stage1 + delta
        _, normal_dot = _project_to_tangent(delta, normal)
        return pc_current, delta, last_score, normal, normal_dot

    def refine(self, pc_stage1, pc_normal_proxy=None, pc_reference=None):
        normal, feat, extra = self._build_context_state(
            pc_stage1,
            pc_reference=pc_reference,
            pc_normal_proxy=pc_normal_proxy,
        )
        return self._refine_with_context(pc_stage1, feat, extra, normal)

    def get_distribution_loss(
        self,
        pc_stage1,
        pc_score_target,
        pc_surface_bank=None,
        pc_normal_proxy=None,
        pc_reference=None,
    ):
        normal, feat, extra = self._build_context_state(
            pc_stage1,
            pc_reference=pc_reference,
            pc_normal_proxy=pc_normal_proxy,
        )
        query = self._sample_queries(pc_stage1, normal)
        pred_score = self._predict_score_from_context(
            query,
            pc_stage1,
            feat,
            extra,
        )
        target_score, target_weight = self._pointset_score_target(query, pc_score_target)
        score_loss = _weighted_mean(
            ((pred_score - target_score) ** 2.0).sum(dim=-1),
            target_weight,
        )

        pc_final, delta, _, normal, normal_dot = self._refine_with_context(
            pc_stage1,
            feat,
            extra,
            normal,
        )

        idx = _random_indices(pc_stage1.shape[1], self.loss_num_points)
        if idx is not None:
            pc_stage1_l = pc_stage1[:, idx, :]
            pc_final_l = pc_final[:, idx, :]
            delta_l = delta[:, idx, :]
            normal_dot_l = normal_dot[:, idx, :]
            pc_reference_l = pc_reference[:, idx, :] if pc_reference is not None else None
        else:
            pc_stage1_l = pc_stage1
            pc_final_l = pc_final
            delta_l = delta
            normal_dot_l = normal_dot
            pc_reference_l = pc_reference

        repulsion_loss = 0.0
        spacing_loss = 0.0
        hole_loss = 0.0
        final_knn = _self_knn_distances(pc_final_l, self.repulsion_k)
        if final_knn is not None:
            reference_spacing = None
            if self.distribution_spacing_source == "score_target":
                reference_spacing = self._target_spacing_at_queries(
                    pc_final_l,
                    pc_score_target,
                    self.repulsion_k,
                )
            elif pc_reference_l is not None:
                normal_l = normal[:, idx, :] if idx is not None else normal
                residual_tan, _ = _project_to_tangent(pc_reference_l - pc_stage1_l, normal_l)
                pc_reference_tangent = pc_stage1_l + residual_tan
                reference_spacing, _, _ = _local_spacing(pc_reference_tangent, self.repulsion_k)
            if reference_spacing is not None:
                final_spacing = final_knn.mean(dim=2)
                target_spacing = reference_spacing * float(self.spacing_target_scale)
                min_radius = target_spacing.unsqueeze(-1) * float(self.repulsion_radius_scale)
                close = jt.maximum(min_radius - final_knn, jt.zeros_like(final_knn))
                repulsion_loss = (close ** 2.0).mean()
                shortfall = jt.maximum(
                    target_spacing - final_spacing,
                    jt.zeros_like(final_spacing),
                )
                spacing_loss = (shortfall ** 2.0).mean()
                upper = reference_spacing * float(self.hole_upper_scale)
                excess = jt.maximum(
                    final_spacing - upper,
                    jt.zeros_like(final_spacing),
                )
                hole_loss = (excess ** 2.0).mean()

        normal_delta = (normal_dot_l.squeeze(-1) ** 2.0).mean()
        anchor = (delta_l ** 2.0).sum(dim=-1).mean()

        surface_bank_guard = 0.0
        if self.surface_bank_guard_weight > 0 and pc_surface_bank is not None:
            surface_bank_guard = _surface_bank_guard_loss(
                pc_stage1=pc_stage1,
                pc_final=pc_final,
                pc_surface_bank=pc_surface_bank,
                num_points=self.surface_bank_guard_num_points,
                bank_num_points=self.surface_bank_guard_bank_points,
                margin=self.surface_bank_guard_margin,
            )

        loss = (
            self.score_loss_weight * score_loss +
            self.repulsion_loss_weight * repulsion_loss +
            self.spacing_loss_weight * spacing_loss +
            self.hole_loss_weight * hole_loss +
            self.normal_delta_weight * normal_delta +
            self.anchor_loss_weight * anchor +
            self.surface_bank_guard_weight * surface_bank_guard
        ) / self.dsm_sigma
        return loss

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None):
        with jt.no_grad():
            pc_stage1 = self._run_stage1(pcl_noisy, num_steps=num_steps)
            pc_final, delta, _, _, _ = self.refine(
                pc_stage1,
                pc_reference=pcl_noisy,
            )
        return pc_final, delta

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_stage1"].shape[-2]
        pc_stage1 = batch["pc_stage1"].reshape(-1, patch_size, 3)

        pc_reference = batch.get("pc_reference", None)
        if pc_reference is not None:
            pc_reference = pc_reference.reshape(-1, patch_size, 3)

        pc_surface_bank = batch.get("pc_surface_bank", None)
        if pc_surface_bank is not None:
            surface_bank_size = pc_surface_bank.shape[-2]
            pc_surface_bank = pc_surface_bank.reshape(-1, surface_bank_size, 3)

        pc_score_target = batch.get("pc_score_target", None)
        if pc_score_target is not None:
            target_size = pc_score_target.shape[-2]
            pc_score_target = pc_score_target.reshape(-1, target_size, 3)
        elif pc_surface_bank is not None:
            pc_score_target = pc_surface_bank
        else:
            raise KeyError(
                "ScoreFieldRefineModule needs pc_surface_bank or pc_score_target "
                "in the refine cache."
            )

        pc_normal_proxy = batch.get("pc_normal_proxy", None)
        if pc_normal_proxy is not None:
            pc_normal_proxy = pc_normal_proxy.reshape(-1, patch_size, 3)

        loss = self.get_distribution_loss(
            pc_stage1=pc_stage1,
            pc_score_target=pc_score_target,
            pc_surface_bank=pc_surface_bank,
            pc_normal_proxy=pc_normal_proxy,
            pc_reference=pc_reference,
        )
        return {"loss": loss}

    def execute(self, **kwargs) -> Dict:  # type: ignore
        return self.training_step(**kwargs)

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
                d = {"pc_stage1": b.meta["pc_stage1"]}
                if self.reference_field in b.meta:
                    d["pc_reference"] = b.meta[self.reference_field]
                if self.surface_bank_field in b.meta:
                    d["pc_surface_bank"] = b.meta[self.surface_bank_field]
                if self.score_target_field in b.meta:
                    d["pc_score_target"] = b.meta[self.score_target_field]
                elif self.score_fallback_target_field in b.meta:
                    d["pc_score_target"] = b.meta[self.score_fallback_target_field]
                if self.normal_field in b.meta:
                    d["pc_normal_proxy"] = b.meta[self.normal_field]
                res.append(d)
            else:
                d = {"pc_noisy": b.sampled_vertices_noisy}
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res


class NoisyAnchorScoreFieldRefineModule(ScoreFieldRefineModule):
    """
    Score-Denoise style refinement conditioned on the noisy point set.

    The encoder and score anchors are pc_noisy. Queries are sampled from
    pc_stage1/current refinement positions. Clean points are only used during
    training to build the score target, never as prediction-time input.
    """

    def __init__(self, model_config, transform_config):
        cfg = deepcopy(model_config)
        cfg.setdefault("score_target_field", "pc_clean_corr")
        cfg.setdefault("score_fallback_target_field", "pc_clean")
        cfg.setdefault("distribution_spacing_source", "noisy")
        cfg.setdefault("anchor_loss_weight", 0.0)
        cfg.setdefault("surface_bank_guard_weight", 0.0)
        super().__init__(cfg, transform_config)

    def _noisy_anchor_extra(self, pc_stage1, pc_noisy, normal):
        B, N, _ = pc_noisy.shape

        stage_spacing, _, _ = _local_spacing(pc_stage1, self.context_feature_k)
        if stage_spacing is None:
            stage_spacing = jt.ones((B, N)) * 1e-3

        noisy_spacing, _, _ = _local_spacing(pc_noisy, self.context_feature_k)
        if noisy_spacing is None:
            noisy_spacing = stage_spacing

        residual = pc_noisy - pc_stage1
        residual_tan, residual_normal = _project_to_tangent(residual, normal)
        residual_norm = jt.sqrt((residual_tan ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
        residual_normal_abs = jt.abs(residual_normal)

        stage_spacing_u = stage_spacing.unsqueeze(-1)
        noisy_spacing_u = noisy_spacing.unsqueeze(-1)
        spacing_ratio = stage_spacing_u / (noisy_spacing_u + 1e-8)
        dense = jt.maximum(
            (noisy_spacing_u * float(self.context_reference_spacing_scale) - stage_spacing_u) /
            (noisy_spacing_u + 1e-8),
            jt.zeros_like(stage_spacing_u),
        )
        sparse = jt.maximum(
            (stage_spacing_u - noisy_spacing_u * float(self.hole_spacing_scale)) /
            (noisy_spacing_u + 1e-8),
            jt.zeros_like(stage_spacing_u),
        )

        return jt.concat(
            [
                residual_tan,
                residual_norm,
                residual_normal_abs,
                stage_spacing_u,
                noisy_spacing_u,
                spacing_ratio,
                dense,
                sparse,
            ],
            dim=-1,
        )

    def _build_noisy_anchor_state(self, pc_stage1, pc_noisy, pc_normal_proxy=None):
        if pc_noisy is None:
            raise KeyError(
                "NoisyAnchorScoreFieldRefineModule requires pc_noisy. "
                "The score field must be conditioned on noisy anchors."
            )
        if pc_noisy.shape[1] != pc_stage1.shape[1]:
            raise ValueError(
                "Noisy-anchor score training expects pc_noisy and pc_stage1 "
                f"to have the same number of points, got {pc_noisy.shape[1]} "
                f"and {pc_stage1.shape[1]}."
            )
        normal = self._normal_proxy(pc_stage1, pc_normal_proxy=pc_normal_proxy)
        feat = self.encoder(pc_noisy)
        extra = self._noisy_anchor_extra(pc_stage1, pc_noisy, normal)
        return normal, feat, extra

    def _refine_with_noisy_context(self, pc_stage1, pc_noisy, feat, extra, normal):
        pc_current = pc_stage1
        last_score = jt.zeros_like(pc_stage1)
        step = float(self.score_step_size)
        for _ in range(int(self.score_steps)):
            delta_step, last_score = self._apply_score_step(
                pc_current,
                pc_noisy,
                feat,
                extra,
                normal,
                step,
            )
            pc_current = pc_current + delta_step
            step *= float(self.score_step_decay)

        delta = pc_current - pc_stage1
        if self.score_total_cap and self.score_total_cap > 0:
            norm = jt.sqrt((delta ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
            delta = delta * jt.minimum(
                jt.ones_like(norm),
                jt.ones_like(norm) * float(self.score_total_cap) / (norm + 1e-8),
            )
            pc_current = pc_stage1 + delta
        _, normal_dot = _project_to_tangent(delta, normal)
        return pc_current, delta, last_score, normal, normal_dot

    def predict_score(self, query, pc_stage1, pc_noisy, pc_normal_proxy=None):
        _, feat, extra = self._build_noisy_anchor_state(
            pc_stage1,
            pc_noisy,
            pc_normal_proxy=pc_normal_proxy,
        )
        return self._predict_score_from_context(query, pc_noisy, feat, extra)

    def refine(self, pc_stage1, pc_noisy=None, pc_normal_proxy=None):
        normal, feat, extra = self._build_noisy_anchor_state(
            pc_stage1,
            pc_noisy,
            pc_normal_proxy=pc_normal_proxy,
        )
        return self._refine_with_noisy_context(pc_stage1, pc_noisy, feat, extra, normal)

    def get_distribution_loss(
        self,
        pc_stage1,
        pc_score_target,
        pc_noisy=None,
        pc_normal_proxy=None,
    ):
        normal, feat, extra = self._build_noisy_anchor_state(
            pc_stage1,
            pc_noisy,
            pc_normal_proxy=pc_normal_proxy,
        )
        query = self._sample_queries(pc_stage1, normal)
        pred_score = self._predict_score_from_context(
            query,
            pc_noisy,
            feat,
            extra,
        )
        target_score, target_weight = self._pointset_score_target(query, pc_score_target)
        score_loss = _weighted_mean(
            ((pred_score - target_score) ** 2.0).sum(dim=-1),
            target_weight,
        )

        pc_final, delta, _, normal, normal_dot = self._refine_with_noisy_context(
            pc_stage1,
            pc_noisy,
            feat,
            extra,
            normal,
        )

        idx = _random_indices(pc_stage1.shape[1], self.loss_num_points)
        if idx is not None:
            pc_final_l = pc_final[:, idx, :]
            delta_l = delta[:, idx, :]
            normal_dot_l = normal_dot[:, idx, :]
            pc_noisy_l = pc_noisy[:, idx, :]
        else:
            pc_final_l = pc_final
            delta_l = delta
            normal_dot_l = normal_dot
            pc_noisy_l = pc_noisy

        repulsion_loss = 0.0
        spacing_loss = 0.0
        hole_loss = 0.0
        final_knn = _self_knn_distances(pc_final_l, self.repulsion_k)
        if final_knn is not None:
            reference_spacing = None
            if self.distribution_spacing_source == "noisy":
                reference_spacing, _, _ = _local_spacing(pc_noisy_l, self.repulsion_k)
            if reference_spacing is not None:
                final_spacing = final_knn.mean(dim=2)
                target_spacing = reference_spacing * float(self.spacing_target_scale)
                min_radius = target_spacing.unsqueeze(-1) * float(self.repulsion_radius_scale)
                close = jt.maximum(min_radius - final_knn, jt.zeros_like(final_knn))
                repulsion_loss = (close ** 2.0).mean()
                shortfall = jt.maximum(
                    target_spacing - final_spacing,
                    jt.zeros_like(final_spacing),
                )
                spacing_loss = (shortfall ** 2.0).mean()
                upper = reference_spacing * float(self.hole_upper_scale)
                excess = jt.maximum(
                    final_spacing - upper,
                    jt.zeros_like(final_spacing),
                )
                hole_loss = (excess ** 2.0).mean()

        normal_delta = (normal_dot_l.squeeze(-1) ** 2.0).mean()
        anchor = (delta_l ** 2.0).sum(dim=-1).mean()

        loss = (
            self.score_loss_weight * score_loss +
            self.repulsion_loss_weight * repulsion_loss +
            self.spacing_loss_weight * spacing_loss +
            self.hole_loss_weight * hole_loss +
            self.normal_delta_weight * normal_delta +
            self.anchor_loss_weight * anchor
        ) / self.dsm_sigma
        return loss

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None):
        with jt.no_grad():
            pc_stage1 = self._run_stage1(pcl_noisy, num_steps=num_steps)
            pc_final, delta, _, _, _ = self.refine(
                pc_stage1,
                pc_noisy=pcl_noisy,
            )
        return pc_final, delta

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_stage1"].shape[-2]
        pc_stage1 = batch["pc_stage1"].reshape(-1, patch_size, 3)
        pc_noisy = batch["pc_noisy"].reshape(-1, patch_size, 3)

        pc_score_target = batch.get("pc_score_target", None)
        if pc_score_target is None:
            raise KeyError(
                "NoisyAnchorScoreFieldRefineModule needs pc_clean_corr/pc_clean "
                "as pc_score_target for training supervision."
            )
        target_size = pc_score_target.shape[-2]
        pc_score_target = pc_score_target.reshape(-1, target_size, 3)

        pc_normal_proxy = batch.get("pc_normal_proxy", None)
        if pc_normal_proxy is not None:
            pc_normal_proxy = pc_normal_proxy.reshape(-1, patch_size, 3)

        loss = self.get_distribution_loss(
            pc_stage1=pc_stage1,
            pc_score_target=pc_score_target,
            pc_noisy=pc_noisy,
            pc_normal_proxy=pc_normal_proxy,
        )
        return {"loss": loss}

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
                if self.reference_field not in b.meta:
                    raise KeyError(
                        f"{b.path} does not contain {self.reference_field}; "
                        "noisy-anchor score training needs pc_noisy as model input."
                    )
                target_key = self.score_target_field
                if target_key not in b.meta:
                    target_key = self.score_fallback_target_field
                if target_key not in b.meta:
                    raise KeyError(
                        f"{b.path} does not contain {self.score_target_field} or "
                        f"{self.score_fallback_target_field} for score supervision."
                    )
                d = {
                    "pc_stage1": b.meta["pc_stage1"],
                    "pc_noisy": b.meta[self.reference_field],
                    "pc_score_target": b.meta[target_key],
                }
                if self.normal_field in b.meta:
                    d["pc_normal_proxy"] = b.meta[self.normal_field]
                res.append(d)
            else:
                d = {"pc_noisy": b.sampled_vertices_noisy}
                res.append(d)
        return res


class TangentialCDRefineModule(CDRefineModule):
    pass


