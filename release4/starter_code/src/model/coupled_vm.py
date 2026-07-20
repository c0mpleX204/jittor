from typing import Dict, List, Optional

import jittor as jt

from .dcvm import DirectionDistanceVelocityModule
from .spec import ModelSpec
from .vm import get_random_indices, patch_based_denoise

from ..data.asset import Asset


class CoupledSurfaceStraightVelocityModule(ModelSpec):
    """
    Coupled Surface-Straight VM.

    Multiple VelocityModules are applied sequentially within each denoising step.
    When all modules are initialized from the same single-VM checkpoint, the
    initial inference trajectory is equivalent to the original VM. Fine-tuning
    then specializes each module to a slightly different segment of the straight
    path.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.num_coupled_vms = cfg.get("num_coupled_vms", 2)
        self.denoise_steps = cfg.get("denoise_steps", 4)
        self.num_train_points = cfg.get("num_train_points", 128)
        self.dsm_sigma = cfg.get("dsm_sigma", 0.01)
        self.velocity_loss_weight = cfg.get("velocity_loss_weight", 1.0)
        self.state_loss_weight = cfg.get("state_loss_weight", 0.25)

        velocity_cfg = cfg.get("velocity_model", None)
        if velocity_cfg is None:
            velocity_cfg = {
                "frame_knn": cfg.get("frame_knn", 16),
                "num_train_points": self.num_train_points,
                "feat_embedding_dim": cfg.get("feat_embedding_dim", 256),
                "decoder_hidden_dim": cfg.get("decoder_hidden_dim", 64),
                "dsm_sigma": self.dsm_sigma,
                "denoise_steps": self.denoise_steps,
            }

        self.vm_names = []
        init_ckpt = cfg.get("init_ckpt", None)
        for i in range(self.num_coupled_vms):
            name = f"velocity_model_{i}"
            vm = DirectionDistanceVelocityModule(
                model_config=velocity_cfg,
                transform_config=transform_config,
            )
            if init_ckpt is not None:
                vm.load(init_ckpt)
            setattr(self, name, vm)
            self.vm_names.append(name)

    def _vms(self):
        return [getattr(self, name) for name in self.vm_names]

    def _predict_velocity(self, vm, pc_state, pnt_idx: Optional[jt.Var]=None):
        B, N, d = pc_state.shape
        feat = vm.encoder(pc_state)
        F_dim = feat.shape[-1]

        if pnt_idx is not None:
            feat = feat[:, pnt_idx, :]
            out_n = len(pnt_idx)
        else:
            out_n = N

        pred_velocity = vm.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, out_n, d)
        return pred_velocity

    def get_supervised_loss(self, pc_noisy_l2, pc_current, pc_surface):
        """
        pc_noisy_l2: high-noise endpoint.
        pc_current: t * pc_surface + (1 - t) * pc_noisy_l2.
        pc_surface: target endpoint prepared by the transform.
        """
        _, N, _ = pc_current.shape
        pnt_idx = get_random_indices(N, self.num_train_points)

        target_velocity = pc_surface - pc_noisy_l2
        target_velocity_sel = target_velocity[:, pnt_idx, :]
        pc_current_sel = pc_current[:, pnt_idx, :]

        total_velocity_loss = 0.0
        total_state_loss = 0.0
        total_substeps = self.denoise_steps * self.num_coupled_vms

        for i, vm in enumerate(self._vms()):
            segment_offset = float(i) / float(total_substeps)
            segment_step = 1.0 / float(total_substeps)
            pc_state = pc_current + segment_offset * target_velocity
            pc_state_sel = pc_state[:, pnt_idx, :]

            pred_velocity = self._predict_velocity(vm, pc_state, pnt_idx=pnt_idx)
            velocity_loss = (((pred_velocity - target_velocity_sel) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
            total_velocity_loss = total_velocity_loss + velocity_loss

            if self.state_loss_weight > 0:
                pred_next = pc_state_sel + segment_step * pred_velocity
                expected_next = pc_current_sel + (segment_offset + segment_step) * target_velocity_sel
                state_loss = (((pred_next - expected_next) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
                total_state_loss = total_state_loss + state_loss

        total_velocity_loss = total_velocity_loss / self.num_coupled_vms
        total_state_loss = total_state_loss / self.num_coupled_vms
        return (
            self.velocity_loss_weight * total_velocity_loss +
            self.state_loss_weight * total_state_loss
        )

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None):
        """
        pcl_noisy: (B, N, 3)
        """
        if num_steps is None:
            num_steps = self.denoise_steps

        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            denom = float(num_steps * self.num_coupled_vms)
            for _ in range(num_steps):
                for vm in self._vms():
                    pred_velocity = self._predict_velocity(vm, pcl_next)
                    pcl_next = pcl_next + (1.0 / denom) * pred_velocity
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_mix"].shape[-2]
        pc_noisy_l2 = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_current = batch["pc_mix"].reshape(-1, patch_size, 3)
        pc_surface = batch["pc_clean"].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_noisy_l2=pc_noisy_l2,
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
