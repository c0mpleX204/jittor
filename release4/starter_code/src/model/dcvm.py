from typing import Dict, List

import jittor as jt
from jittor import nn

from .feature import Decoder, FeatureExtraction
from .spec import ModelSpec
from .vm import get_random_indices, patch_based_denoise

from ..data.asset import Asset


def _clamp01(x):
    return jt.minimum(jt.maximum(x, jt.zeros_like(x)), jt.ones_like(x))


def _weighted_mean(value, weight):
    return (value * weight).sum() / (weight.sum() + 1e-6)


class DirectionDistanceVelocityModule(ModelSpec):
    """
    Surface-Straight VM.

    This replaces the failed v1 gated single-step delta with the VM stage from
    StraightPCF: features are extracted from an interpolated current patch, but
    the target is the fixed straight velocity from the high-noise endpoint to
    the surface endpoint prepared by the transform.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.frame_knn = cfg["frame_knn"]
        self.num_train_points = cfg["num_train_points"]
        self.dsm_sigma = cfg["dsm_sigma"]
        self.denoise_steps = cfg.get("denoise_steps", 4)
        self.edge_velocity_anchor_weight = cfg.get("edge_velocity_anchor_weight", 0.0)

        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg["feat_embedding_dim"],
        )

        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg["decoder_hidden_dim"],
        )

    def get_supervised_loss(self, pc_noisy_l2, pc_current, pc_surface, pc_edge_risk=None):
        """
        pc_noisy_l2: high-noise endpoint, equivalent to StraightPCF pcl_noisy_L2.
        pc_current: t * pc_surface + (1 - t) * pc_noisy_l2.
        pc_surface: nearest sampled surface endpoint.
        """
        B, N, d = pc_current.shape
        pnt_idx = get_random_indices(N, self.num_train_points)

        feat = self.encoder(pc_current)
        F_dim = feat.shape[-1]
        feat = feat[:, pnt_idx, :]
        pc_noisy_l2 = pc_noisy_l2[:, pnt_idx, :]
        pc_surface = pc_surface[:, pnt_idx, :]
        if pc_edge_risk is not None:
            pc_edge_risk = pc_edge_risk[:, pnt_idx, :]

        target_velocity = pc_surface - pc_noisy_l2
        pred_velocity = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d)

        dir_loss = ((pred_velocity - target_velocity) ** 2.0).sum(dim=-1).mean()
        edge_anchor = 0.0
        if pc_edge_risk is not None and self.edge_velocity_anchor_weight > 0:
            edge_anchor = (pc_edge_risk.squeeze(-1) * (pred_velocity ** 2.0).sum(dim=-1)).mean()
        return (dir_loss + self.edge_velocity_anchor_weight * edge_anchor) / self.dsm_sigma

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None):
        """
        pcl_noisy: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        if num_steps is None:
            num_steps = self.denoise_steps

        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for _ in range(num_steps):
                feat = self.encoder(pcl_next)
                F_dim = feat.shape[-1]
                pred_velocity = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)
                pcl_next = pcl_next + (1.0 / num_steps) * pred_velocity
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_mix"].shape[-2]
        pc_noisy_l2 = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_current = batch["pc_mix"].reshape(-1, patch_size, 3)
        pc_surface = batch["pc_clean"].reshape(-1, patch_size, 3)
        pc_edge_risk = batch.get("pc_edge_risk", None)
        if pc_edge_risk is not None:
            pc_edge_risk = pc_edge_risk.reshape(-1, patch_size, 1)
        loss = self.get_supervised_loss(
            pc_noisy_l2=pc_noisy_l2,
            pc_current=pc_current,
            pc_surface=pc_surface,
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
                if "pc_edge_risk" in b.meta:
                    d["pc_edge_risk"] = b.meta["pc_edge_risk"]
                res.append(d)
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy,
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res


class EdgeAwareDirectionDistanceVelocityModule(DirectionDistanceVelocityModule):
    """
    Surface-Straight VM with an edge/thin-structure protection gate.

    This keeps DirectionDistanceVelocityModule's training target and inference
    dynamics unchanged for ordinary surfaces. High edge-risk points are moved
    less in stage 1, leaving the detailed correction to later refinement stages.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.gate_head = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, cfg["decoder_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["decoder_hidden_dim"], 1),
        )

        self.gate_bias = cfg.get("gate_bias", -4.0)
        self.edge_protect_strength = cfg.get("edge_protect_strength", 0.25)
        self.base_head_loss_weight = cfg.get("base_head_loss_weight", 0.10)
        self.gate_loss_weight = cfg.get("gate_loss_weight", 0.03)
        self.gate_sparsity_weight = cfg.get("gate_sparsity_weight", 0.01)
        self.gate_target_power = cfg.get("gate_target_power", 1.2)

    def _decode_velocity(self, feat, B, N, d):
        F_dim = feat.shape[-1]
        feat_flat = feat.reshape(-1, F_dim)
        base_velocity = self.decoder(
            c=feat_flat,
        ).reshape(B, N, d)
        gate = jt.sigmoid(
            self.gate_head(feat_flat).reshape(B, N, 1) + float(self.gate_bias)
        )
        scale = 1.0 - float(self.edge_protect_strength) * gate
        pred_velocity = scale * base_velocity
        return pred_velocity, base_velocity, gate

    def _predict_velocity(self, pc_current):
        B, N, d = pc_current.shape
        feat = self.encoder(pc_current)
        return self._decode_velocity(feat, B, N, d)

    def get_supervised_loss(self, pc_noisy_l2, pc_current, pc_surface, pc_edge_risk=None):
        B, N, d = pc_current.shape
        pnt_idx = get_random_indices(N, self.num_train_points)

        feat = self.encoder(pc_current)
        feat = feat[:, pnt_idx, :]
        pc_noisy_l2 = pc_noisy_l2[:, pnt_idx, :]
        pc_surface = pc_surface[:, pnt_idx, :]
        if pc_edge_risk is not None:
            pc_edge_risk = pc_edge_risk[:, pnt_idx, :]

        target_velocity = pc_surface - pc_noisy_l2
        pred_velocity, base_velocity, gate = self._decode_velocity(
            feat=feat,
            B=B,
            N=len(pnt_idx),
            d=d,
        )

        if pc_edge_risk is None:
            main_mse = ((pred_velocity - target_velocity) ** 2.0).sum(dim=-1)
            loss = main_mse.mean() / self.dsm_sigma
            return loss

        edge_risk = _clamp01(pc_edge_risk)
        normal_weight = 1.0 - edge_risk

        gate_target = edge_risk
        if self.gate_target_power != 1.0:
            gate_target = gate_target ** float(self.gate_target_power)

        target_scale = 1.0 - float(self.edge_protect_strength) * gate_target
        protected_target_velocity = target_scale * target_velocity
        main_mse = ((pred_velocity - protected_target_velocity) ** 2.0).sum(dim=-1)
        loss = main_mse.mean() / self.dsm_sigma

        base_mse = ((base_velocity - target_velocity) ** 2.0).sum(dim=-1, keepdims=True)
        base_head_loss = _weighted_mean(base_mse / self.dsm_sigma, normal_weight)
        gate_loss = ((gate - gate_target) ** 2.0).mean()
        gate_sparsity = (normal_weight * (gate ** 2.0)).mean()

        edge_anchor = 0.0
        if self.edge_velocity_anchor_weight > 0:
            edge_anchor = (edge_risk.squeeze(-1) * (pred_velocity ** 2.0).sum(dim=-1)).mean()

        return (
            loss
            + self.base_head_loss_weight * base_head_loss
            + self.gate_loss_weight * gate_loss
            + self.gate_sparsity_weight * gate_sparsity
            + (self.edge_velocity_anchor_weight * edge_anchor) / self.dsm_sigma
        )

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None):
        B, N, d = pcl_noisy.shape
        if num_steps is None:
            num_steps = self.denoise_steps

        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for _ in range(num_steps):
                pred_velocity, _, _ = self._predict_velocity(pcl_next)
                pcl_next = pcl_next + (1.0 / num_steps) * pred_velocity
        return pcl_next, None
