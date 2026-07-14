from typing import Dict, List

import jittor as jt
from jittor import nn

from .feature import FeatureExtraction
from .spec import ModelSpec
from .vm import get_random_indices, patch_based_denoise

from ..data.asset import Asset


class PointDecoder(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
        )

    def execute(self, c):
        return self.net(c)


class DirectionDistanceVelocityModule(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.frame_knn = cfg["frame_knn"]
        self.num_train_points = cfg["num_train_points"]
        self.dsm_sigma = cfg["dsm_sigma"]

        self.velocity_max = cfg.get("velocity_max", 0.15)
        self.predict_step_scale = cfg.get("predict_step_scale", 1.0)

        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg["feat_embedding_dim"],
        )

        self.decoder = PointDecoder(
            z_dim=self.encoder.embedding_dim,
            out_dim=4,
            hidden_size=cfg["decoder_hidden_dim"],
        )

    def _predict_delta_from_feat(self, feat, B: int, N: int):
        F_dim = feat.shape[-1]
        pred = self.decoder(feat.reshape(-1, F_dim)).reshape(B, N, 4)
        velocity = jt.tanh(pred[:, :, :3]) * self.velocity_max
        gate = jt.sigmoid(pred[:, :, 3:4])
        pred_delta = velocity * gate
        return pred_delta

    def get_supervised_loss(self, pc_current, pc_clean):
        """
        Learn a single bounded displacement from the current noisy/mixed state
        toward the clean surface.
        """
        B, N, _ = pc_current.shape
        pnt_idx = get_random_indices(N, self.num_train_points)

        feat = self.encoder(pc_current)
        feat = feat[:, pnt_idx, :]
        pc_current = pc_current[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]

        target_delta = pc_clean - pc_current
        pred_delta = self._predict_delta_from_feat(
            feat=feat,
            B=B,
            N=len(pnt_idx),
        )

        return (((pred_delta - target_delta) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=1):
        """
        pcl_noisy: (B, N, 3)
        """
        B, N, _ = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for _ in range(num_steps):
                feat = self.encoder(pcl_next)
                pred_delta = self._predict_delta_from_feat(feat=feat, B=B, N=N)
                pcl_next = pcl_next + (self.predict_step_scale / num_steps) * pred_delta
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_mix"].shape[-2]
        pc_mix = batch["pc_mix"].reshape(-1, patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_current=pc_mix,
            pc_clean=pc_clean,
        )
        return {"loss": loss}

    def execute(self, **kwargs) -> Dict: # type: ignore
        return self.training_step(**kwargs)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch["pc_noisy"]
        assert pc_noisy_batch.ndim == 3

        res = []
        for pc_noisy in pc_noisy_batch:
            pc_next = patch_based_denoise(
                model=self, # type: ignore[arg-type]
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
                res.append({
                    "pc_clean": b.meta["pc_clean"],
                    "pc_mix": b.meta["pc_mix"],
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy,
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res
