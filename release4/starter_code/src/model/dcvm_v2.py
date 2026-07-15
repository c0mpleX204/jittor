import jittor as jt

from .dcvm import DirectionDistanceVelocityModule
from .vm import get_random_indices


class SurfaceTargetVelocityModule(DirectionDistanceVelocityModule):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.surface_loss_weight = cfg.get("surface_loss_weight", 1.0)
        self.paired_loss_weight = cfg.get("paired_loss_weight", 0.1)

    def _nearest_clean_points(self, pc_query, pc_clean):
        """
        For each current point, use the nearest clean point in the same patch as
        a discrete surface target. This removes the strict same-index target.
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
