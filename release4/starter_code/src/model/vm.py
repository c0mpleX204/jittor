from math import ceil
from typing import Dict, List

import jittor as jt
import numpy as np
from jittor import nn

from .feature import FeatureExtraction, Decoder
from .spec import ModelSpec

from ..data.asset import Asset

def get_random_indices(n, m):
    assert m < n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()

def _clamp01(x):
    return jt.minimum(jt.maximum(x, jt.zeros_like(x)), jt.ones_like(x))

def _weighted_mean(value, weight):
    return (value * weight).sum() / (weight.sum() + 1e-6)

class VelocityModule(ModelSpec):
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        # geometry
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']
        
        # score-matching
        self.dsm_sigma = cfg['dsm_sigma']
        
        # networks
        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg['feat_embedding_dim']
        )
        
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )
    
    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        """
        pcl_noisy: (B, N, 3)
        pcl_clean: (B, N, 3)
        """
        B, N_noisy, d = pc_mix.shape
        
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)
        
        # Feature extraction
        feat = self.encoder(pc_mix)  # (B, N, F)
        F_dim = feat.shape[2]
        
        # gather
        feat = feat[:, pnt_idx, :]
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]
        
        # target
        grad_dir_t_target = pc_clean - pc_noisy
        
        # decoder
        pred_dir = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d) # type: ignore
        
        loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        
        return loss

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=4):
        """
        pcl_noisy: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for it in range(num_steps):
                feat = self.encoder(pcl_next)  # (B, N, F)
                F_dim = feat.shape[2]
                
                pred_dir = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)
                
                pcl_next = pcl_next + (1.0 / num_steps) * pred_dir
        return pcl_next, None
    
    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
        )
        return {"loss": loss}
    
    def execute(self, **kwargs) -> Dict: # type: ignore
        return self.training_step(**kwargs)
    
    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        
        num_steps = 1
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_next = pc_noisy
            for it in range(num_steps):
                pc_next = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
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
                    "pc_noisy": b.meta['pc_noisy'], # (num_patches, patch_size, 3)
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy, # (N, 3)
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res

class EdgeAwareVelocityModule(VelocityModule):
    """
    VM with a gated edge/thin-structure branch.

    The smooth branch keeps the original VM behavior. The edge branch receives
    extra supervision on high-risk points, while the gate is weakly trained from
    pc_edge_risk and initialized low so ordinary surfaces stay on the smooth path.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.denoise_steps = cfg.get('denoise_steps', 4)
        self.target_field = cfg.get('target_field', 'pc_clean')
        self.fallback_target_field = cfg.get('fallback_target_field', 'pc_clean')

        self.edge_decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )
        self.gate_head = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, cfg['decoder_hidden_dim']),
            nn.ReLU(),
            nn.Linear(cfg['decoder_hidden_dim'], 1),
        )

        self.gate_bias = cfg.get('gate_bias', -2.0)
        self.edge_loss_weight = cfg.get('edge_loss_weight', 0.5)
        self.smooth_head_loss_weight = cfg.get('smooth_head_loss_weight', 0.15)
        self.edge_head_loss_weight = cfg.get('edge_head_loss_weight', 0.3)
        self.gate_loss_weight = cfg.get('gate_loss_weight', 0.02)
        self.gate_sparsity_weight = cfg.get('gate_sparsity_weight', 0.02)
        self.edge_risk_power = cfg.get('edge_risk_power', 1.0)
        self.gate_target_power = cfg.get('gate_target_power', 1.0)

    def _predict_dir(self, pc_mix):
        B, N, d = pc_mix.shape
        feat = self.encoder(pc_mix)
        F_dim = feat.shape[2]
        feat_flat = feat.reshape(-1, F_dim)

        smooth_dir = self.decoder(
            c=feat_flat,
        ).reshape(B, N, d)
        edge_dir = self.edge_decoder(
            c=feat_flat,
        ).reshape(B, N, d)
        gate = jt.sigmoid(
            self.gate_head(feat_flat).reshape(B, N, 1) + float(self.gate_bias)
        )
        pred_dir = (1.0 - gate) * smooth_dir + gate * edge_dir
        return pred_dir, smooth_dir, edge_dir, gate

    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean, pc_edge_risk=None):
        B, N_noisy, d = pc_mix.shape
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]
        if pc_edge_risk is not None:
            pc_edge_risk = pc_edge_risk[:, pnt_idx, :]

        target = pc_clean - pc_noisy
        pred_dir, smooth_dir, edge_dir, gate = self._predict_dir(pc_mix)

        main_mse = ((pred_dir - target) ** 2.0).sum(dim=-1)
        loss = (main_mse / self.dsm_sigma).mean()

        if pc_edge_risk is None:
            return loss

        edge_risk = _clamp01(pc_edge_risk)
        if self.edge_risk_power != 1.0:
            edge_weight = edge_risk ** float(self.edge_risk_power)
        else:
            edge_weight = edge_risk
        normal_weight = 1.0 - edge_risk

        edge_mse = ((pred_dir - target) ** 2.0).sum(dim=-1, keepdims=True)
        smooth_mse = ((smooth_dir - target) ** 2.0).sum(dim=-1, keepdims=True)
        edge_head_mse = ((edge_dir - target) ** 2.0).sum(dim=-1, keepdims=True)

        edge_loss = _weighted_mean(edge_mse / self.dsm_sigma, edge_weight)
        smooth_head_loss = _weighted_mean(smooth_mse / self.dsm_sigma, normal_weight)
        edge_head_loss = _weighted_mean(edge_head_mse / self.dsm_sigma, edge_weight)

        gate_target = edge_risk
        if self.gate_target_power != 1.0:
            gate_target = gate_target ** float(self.gate_target_power)
        gate_loss = ((gate - gate_target) ** 2.0).mean()
        gate_sparsity = (normal_weight * (gate ** 2.0)).mean()

        return (
            loss +
            self.edge_loss_weight * edge_loss +
            self.smooth_head_loss_weight * smooth_head_loss +
            self.edge_head_loss_weight * edge_head_loss +
            self.gate_loss_weight * gate_loss +
            self.gate_sparsity_weight * gate_sparsity
        )

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None):
        B, N, d = pcl_noisy.shape
        if num_steps is None:
            num_steps = self.denoise_steps
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for it in range(num_steps):
                pred_dir, _, _, _ = self._predict_dir(pcl_next)
                pcl_next = pcl_next + (1.0 / num_steps) * pred_dir
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        pc_edge_risk = batch.get('pc_edge_risk', None)
        if pc_edge_risk is not None:
            pc_edge_risk = pc_edge_risk.reshape(-1, patch_size, 1)
        loss = self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
            pc_edge_risk=pc_edge_risk,
        )
        return {"loss": loss}

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                target_key = self.target_field
                if target_key not in b.meta:
                    target_key = self.fallback_target_field
                if target_key not in b.meta:
                    raise KeyError(
                        f"{b.path} does not contain {self.target_field} or "
                        f"{self.fallback_target_field} for EdgeAwareVelocity supervision."
                    )
                d = {
                    "pc_noisy": b.meta['pc_noisy'],
                    "pc_clean": b.meta[target_key],
                    "pc_mix": b.meta['pc_mix'],
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

def farthest_point_sampling(pcls, num_pnts):
    """
    pcls: (B, N, 3)
    return:
        sampled: (B, num_pnts, 3)
        indices: (B, num_pnts)
    """
    B, N, _ = pcls.shape
    sampled = []
    indices = []
    for b in range(B):
        pts = pcls[b]  # (N, 3)
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for i in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]  # (3,)
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32()
        sampled.append(pts[idx][None, ...])
        indices.append(idx[None, ...])
    sampled = jt.concat(sampled, dim=0)
    indices = jt.concat(indices, dim=0)
    return sampled, indices

def knn_points(x, y, k):
    """
    x: (B, P, 3)
    y: (B, N, 3)
    return:
        dist: (B, P, k)
        idx:  (B, P, k)
        nn:   (B, P, k, 3)
    """
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    B = x.shape[0]
    nn = []
    for b in range(B):
        nn.append(y[b][idx[b]])
    nn = jt.stack(nn, dim=0)
    return dist_k, idx, nn

def patch_based_denoise(
    model: VelocityModule,
    pcl_noisy,
    patch_size=1000,
    seed_k=6,
    seed_k_alpha=1,
    aggregation: str="best",
    weight_temperature: float=1.0,
    weight_floor: float=1e-12,
) -> jt.Var:
    """
    pcl_noisy: (N, 3)
    """
    assert len(pcl_noisy.shape) == 2
    
    N, d = pcl_noisy.shape
    num_patches = int(seed_k * N / patch_size)
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    
    seed_pnts, seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)
    
    patches = patches[0]              # (P, M, 3)
    patch_dists = patch_dists[0]      # (P, M)
    point_idxs = point_idxs[0]        # (P, M)
    
    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand
    
    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
    
    all_dists = jt.ones((num_patches, N)) * 1e10
    
    for i in range(num_patches):
        all_dists[i][point_idxs[i]] = patch_dists[i]
        
    patches_denoised = []
    
    i = 0
    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    while i < num_patches:
        curr = patches[i:i+patch_step]
        try:
            out, _ = model.denoise_langevin_dynamics(curr)
        except Exception as e:
            print("Denoise error:", e)
            return None
        patches_denoised.append(out)
        i += patch_step
    
    patches_denoised = jt.concat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_expand

    if aggregation == "weighted":
        temp = max(float(weight_temperature), 1e-6)
        weighted_sum = pcl_noisy[0] * float(weight_floor)
        weight_sum = jt.ones((N, 1)) * float(weight_floor)
        for i in range(num_patches):
            idx = point_idxs[i]
            patch_weights = jt.exp(-patch_dists[i] / temp).unsqueeze(1)
            weighted = patches_denoised[i] * patch_weights
            weighted_sum = weighted_sum.scatter_(
                0,
                idx.unsqueeze(1).broadcast(weighted.shape),
                weighted,
                reduce='add',
            )
            weight_sum = weight_sum.scatter_(
                0,
                idx.unsqueeze(1),
                patch_weights,
                reduce='add',
            )
        pcl_out = weighted_sum / (weight_sum + 1e-12)
        assert pcl_out.shape[0] == N, f"denoised point count mismatch: {pcl_out.shape[0]} != {N}"
        return pcl_out

    if aggregation == "fast_best":
        weights = jt.exp(-all_dists)
        best_weights_idx, _ = jt.argmax(weights, dim=0)

        pcl_out = pcl_noisy[0] * float(weight_floor)
        hit_count = jt.ones((N, 1)) * float(weight_floor)
        for i in range(num_patches):
            idx = point_idxs[i]
            chosen = (best_weights_idx[idx] == i)
            chosen = chosen.float32().unsqueeze(1)
            selected = patches_denoised[i] * chosen
            pcl_out = pcl_out.scatter_(
                0,
                idx.unsqueeze(1).broadcast(selected.shape),
                selected,
                reduce='add',
            )
            hit_count = hit_count.scatter_(
                0,
                idx.unsqueeze(1),
                chosen,
                reduce='add',
            )
        pcl_out = pcl_out / (hit_count + 1e-12)
        assert pcl_out.shape[0] == N, f"denoised point count mismatch: {pcl_out.shape[0]} != {N}"
        return pcl_out

    if aggregation != "best":
        raise ValueError(f"unsupported patch aggregation: {aggregation}")

    weights = jt.exp(-all_dists)
    best_weights_idx, _ = jt.argmax(weights, dim=0)
    pcl_out = []
    for pidx in range(N):
        patch_id = best_weights_idx[pidx].item()
        mask = (point_idxs[patch_id] == pidx)
        if mask.sum().item() > 0:
            pcl_out.append(patches_denoised[patch_id][mask][:1])
        else:
            # Rarely, FPS+KNN patches do not cover every point. Keep the original
            # noisy point so inference always preserves the competition point count.
            pcl_out.append(pcl_noisy[0, pidx:pidx+1])
    pcl_out = jt.concat(pcl_out, dim=0)
    assert pcl_out.shape[0] == N, f"denoised point count mismatch: {pcl_out.shape[0]} != {N}"
    return pcl_out
