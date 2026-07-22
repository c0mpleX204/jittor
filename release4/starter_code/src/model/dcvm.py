from typing import Dict, List

import jittor as jt

from .feature import Decoder, FeatureExtraction
from .spec import ModelSpec
from .vm import get_random_indices, patch_based_denoise

from ..data.asset import Asset


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
