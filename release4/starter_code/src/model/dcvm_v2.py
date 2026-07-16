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


class SurfaceTargetVelocityModule(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.frame_knn = cfg["frame_knn"]
        self.num_train_points = cfg["num_train_points"]
        self.dsm_sigma = cfg["dsm_sigma"]

        self.velocity_max = cfg.get("velocity_max", 0.15)
        self.predict_step_scale = cfg.get("predict_step_scale", 1.0)
        self.surface_loss_weight = cfg.get("surface_loss_weight", 1.0)
        self.paired_loss_weight = cfg.get("paired_loss_weight", 0.1)

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
        return velocity * gate

    def _nearest_clean_points(self, pc_query, pc_clean):
        """
        Use the nearest clean point in the same patch as a discrete surface
        target, instead of forcing same-index point correspondence.
        """
        B, Q, _ = pc_query.shape
        dist = ((pc_query.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        _, nn_idx = jt.topk(dist, k=1, dim=-1, largest=False)
        nn_idx = nn_idx.reshape(B, Q)

        nearest = []
        for b in range(B):
            nearest.append(pc_clean[b][nn_idx[b]][None, ...])
        return jt.concat(nearest, dim=0)

    def _delta_loss(self, pred_delta, target_delta):
        return (((pred_delta - target_delta) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()

    def get_supervised_loss(self, pc_current, pc_clean):
        """
        Learn a one-step displacement toward the nearest clean surface sample,
        with a small paired-point regularizer to preserve point distribution.
        """
        B, N, _ = pc_current.shape
        pnt_idx = get_random_indices(N, self.num_train_points)

        feat = self.encoder(pc_current)
        feat = feat[:, pnt_idx, :]
        pc_query = pc_current[:, pnt_idx, :]
        pc_paired_clean = pc_clean[:, pnt_idx, :]

        pc_surface_clean = self._nearest_clean_points(
            pc_query=pc_query,
            pc_clean=pc_clean,
        )
        surface_delta = pc_surface_clean - pc_query

        pred_delta = self._predict_delta_from_feat(
            feat=feat,
            B=B,
            N=len(pnt_idx),
        )

        loss = self.surface_loss_weight * self._delta_loss(pred_delta, surface_delta)
        if self.paired_loss_weight > 0:
            paired_delta = pc_paired_clean - pc_query
            loss += self.paired_loss_weight * self._delta_loss(pred_delta, paired_delta)
        return loss

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=1):
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
