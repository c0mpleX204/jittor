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
        self.lin_1 = nn.Linear(z_dim, z_dim)
        self.bn_1_out = nn.BatchNorm1d(z_dim)

        self.lin_2 = nn.Linear(z_dim, hidden_size)
        self.bn_2_out = nn.BatchNorm1d(hidden_size)

        self.lin_3 = nn.Linear(hidden_size, out_dim)

        self.actvn_out = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def execute(self, c):
        net = self.lin_1(c)
        net = self.bn_1_out(net)
        net = self.actvn_out(net)
        net = self.dropout(net)

        net = self.lin_2(net)
        net = self.bn_2_out(net)
        net = self.actvn_out(net)
        net = self.dropout(net)

        return self.lin_3(net)


class DirectionDistanceVelocityModule(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.frame_knn = cfg["frame_knn"]
        self.num_train_points = cfg["num_train_points"]
        self.dsm_sigma = cfg["dsm_sigma"]
        self.eps = cfg.get("eps", 1e-8)

        self.delta_loss_weight = cfg.get("delta_loss_weight", 1.0)
        self.direction_loss_weight = cfg.get("direction_loss_weight", 0.1)
        self.distance_loss_weight = cfg.get("distance_loss_weight", 0.2)
        self.predict_step_scale = cfg.get("predict_step_scale", 1.0)

        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg["feat_embedding_dim"],
        )

        self.direction_decoder = PointDecoder(
            z_dim=self.encoder.embedding_dim,
            out_dim=3,
            hidden_size=cfg["decoder_hidden_dim"],
        )
        self.distance_decoder = PointDecoder(
            z_dim=self.encoder.embedding_dim,
            out_dim=1,
            hidden_size=cfg["decoder_hidden_dim"],
        )

    def _normalize_direction(self, raw_direction):
        norm = jt.sqrt((raw_direction ** 2.0).sum(dim=-1, keepdims=True) + self.eps)
        return raw_direction / norm

    def _positive_distance(self, raw_distance):
        return jt.sqrt(raw_distance ** 2.0 + self.eps)

    def _predict_delta_from_feat(self, feat, B: int, N: int):
        F_dim = feat.shape[-1]
        feat_flat = feat.reshape(-1, F_dim)

        pred_direction = self._normalize_direction(
            self.direction_decoder(feat_flat).reshape(B, N, 3)
        )
        pred_distance = self._positive_distance(
            self.distance_decoder(feat_flat).reshape(B, N, 1)
        )
        pred_delta = pred_direction * pred_distance
        return pred_delta, pred_direction, pred_distance

    def get_supervised_loss(self, pc_current, pc_clean):
        """
        Learn the remaining displacement from the current noisy/mixed state to
        the clean surface as direction * distance.
        """
        B, N, _ = pc_current.shape
        pnt_idx = get_random_indices(N, self.num_train_points)

        feat = self.encoder(pc_current)
        feat = feat[:, pnt_idx, :]
        pc_current = pc_current[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]

        target_delta = pc_clean - pc_current
        target_distance = jt.sqrt((target_delta ** 2.0).sum(dim=-1, keepdims=True) + self.eps)
        target_direction = target_delta / target_distance

        pred_delta, pred_direction, pred_distance = self._predict_delta_from_feat(
            feat=feat,
            B=B,
            N=len(pnt_idx),
        )

        delta_loss = (((pred_delta - target_delta) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        direction_loss = (1.0 - (pred_direction * target_direction).sum(dim=-1)).mean()
        distance_loss = (((pred_distance - target_distance) ** 2.0) / self.dsm_sigma).mean()

        loss = (
            self.delta_loss_weight * delta_loss
            + self.direction_loss_weight * direction_loss
            + self.distance_loss_weight * distance_loss
        )
        return loss

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=1):
        """
        pcl_noisy: (B, N, 3)
        """
        B, N, _ = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for _ in range(num_steps):
                feat = self.encoder(pcl_next)
                pred_delta, _, _ = self._predict_delta_from_feat(feat=feat, B=B, N=N)
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
