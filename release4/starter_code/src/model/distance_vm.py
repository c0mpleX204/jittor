from typing import Dict, List, Optional

import jittor as jt

from .dcvm import DirectionDistanceVelocityModule
from .feature import Decoder, FeatureExtraction
from .spec import ModelSpec
from .vm import patch_based_denoise

from ..data.asset import Asset


class StraightPCFDistanceVelocityModule(ModelSpec):
    """
    Patch-level distance-ratio stage on top of a frozen Surface-Straight VM.

    The frozen VM supplies a per-point displacement direction/magnitude. This
    module only predicts one scalar ratio per patch, then applies:
        pc_denoised = pc_current + ratio_patch * vm_displacement
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        velocity_cfg = cfg.get("velocity_model", None)
        if velocity_cfg is None:
            velocity_cfg = {
                "frame_knn": cfg.get("frame_knn", 16),
                "num_train_points": cfg.get("num_train_points", 128),
                "dsm_sigma": cfg.get("dsm_sigma", 0.01),
                "feat_embedding_dim": cfg.get("feat_embedding_dim", 256),
                "decoder_hidden_dim": cfg.get("decoder_hidden_dim", 64),
                "denoise_steps": cfg.get("velocity_denoise_steps", cfg.get("denoise_steps", 4)),
            }
        self.velocity_model = DirectionDistanceVelocityModule(
            model_config=velocity_cfg,
            transform_config=transform_config,
        )
        self.velocity_denoise_steps = cfg.get(
            "velocity_denoise_steps",
            velocity_cfg.get("denoise_steps", 4),
        )

        self.vm_ckpt = cfg.get("vm_ckpt", cfg.get("velocity_ckpt", None))
        if self.vm_ckpt is not None:
            self.velocity_model.load(self.vm_ckpt)
        self._freeze_velocity_model()

        self.frame_knn = cfg.get("distance_frame_knn", cfg.get("frame_knn", 16))
        self.dsm_sigma = cfg.get("dsm_sigma", 0.01)
        self.ratio_min = cfg.get("ratio_min", 0.5)
        self.ratio_max = cfg.get("ratio_max", 1.5)
        self.target_ratio_min = cfg.get("target_ratio_min", self.ratio_min)
        self.target_ratio_max = cfg.get("target_ratio_max", self.ratio_max)
        self.ratio_loss_weight = cfg.get("ratio_loss_weight", 1.0)
        self.reconstruction_loss_weight = cfg.get("reconstruction_loss_weight", 1.0)

        distance_embedding_dim = cfg.get(
            "distance_feat_embedding_dim",
            cfg.get("feat_embedding_dim", 256),
        )
        self.distance_encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=distance_embedding_dim,
            distance_estimation=True,
        )
        self.distance_decoder = Decoder(
            z_dim=self.distance_encoder.embedding_dim,
            dim=3,
            out_dim=1,
            hidden_size=cfg.get("distance_decoder_hidden_dim", cfg.get("decoder_hidden_dim", 64)),
        )

    def _freeze_velocity_model(self):
        self.velocity_model.eval()
        for p in self.velocity_model.parameters():
            if hasattr(p, "stop_grad"):
                p.stop_grad()
            if hasattr(p, "requires_grad"):
                p.requires_grad = False

    def _clip_ratio(self, ratio, min_value: Optional[float], max_value: Optional[float]):
        if min_value is not None:
            ratio = jt.maximum(ratio, jt.ones_like(ratio) * min_value)
        if max_value is not None:
            ratio = jt.minimum(ratio, jt.ones_like(ratio) * max_value)
        return ratio

    def _vm_displacement(self, pc_current, num_steps: int=None):
        if num_steps is None:
            num_steps = self.velocity_denoise_steps
        self.velocity_model.eval()
        with jt.no_grad():
            pc_vm, _ = self.velocity_model.denoise_langevin_dynamics(
                pc_current,
                num_steps=num_steps,
            )
            vm_displacement = pc_vm - pc_current
            if hasattr(vm_displacement, "stop_grad"):
                vm_displacement = vm_displacement.stop_grad()
        return vm_displacement

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

    def get_target_ratio(self, pc_current, pc_surface, vm_displacement):
        target_displacement = pc_surface - pc_current
        numerator = (vm_displacement * target_displacement).sum(dim=-1).sum(dim=-1, keepdims=True)
        denominator = (vm_displacement * vm_displacement).sum(dim=-1).sum(dim=-1, keepdims=True)
        ratio = (numerator / (denominator + 1e-8)).unsqueeze(-1)
        ratio = self._clip_ratio(ratio, self.target_ratio_min, self.target_ratio_max)
        if hasattr(ratio, "stop_grad"):
            ratio = ratio.stop_grad()
        return ratio

    def get_supervised_loss(self, pc_current, pc_surface):
        vm_displacement = self._vm_displacement(pc_current)
        target_ratio = self.get_target_ratio(
            pc_current=pc_current,
            pc_surface=pc_surface,
            vm_displacement=vm_displacement,
        )
        pred_ratio = self.predict_ratio(pc_current)

        ratio_loss = ((pred_ratio - target_ratio) ** 2.0).mean()
        pc_pred = pc_current + pred_ratio.broadcast(pc_current.shape) * vm_displacement
        reconstruction_loss = (((pc_pred - pc_surface) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        return self.ratio_loss_weight * ratio_loss + self.reconstruction_loss_weight * reconstruction_loss

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None):
        if num_steps is None:
            num_steps = self.velocity_denoise_steps
        with jt.no_grad():
            vm_displacement = self._vm_displacement(pcl_noisy, num_steps=num_steps)
            ratio = self.predict_ratio(pcl_noisy)
            pcl_next = pcl_noisy + ratio.broadcast(pcl_noisy.shape) * vm_displacement
        return pcl_next, ratio

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_mix"].shape[-2]
        pc_current = batch["pc_mix"].reshape(-1, patch_size, 3)
        pc_surface = batch["pc_clean"].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_current=pc_current,
            pc_surface=pc_surface,
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
                    "pc_noisy": b.meta["pc_noisy"],
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
