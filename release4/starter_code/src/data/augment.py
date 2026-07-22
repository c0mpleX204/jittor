from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .asset import Asset
from .spec import ConfigSpec
from .utils import random_euler_rotation, sample_vertex_groups


def _as_tuple(x):
    if x is None:
        return None
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def _choose_noise_type(noise_types, noise_probs=None) -> str:
    noise_types = _as_tuple(noise_types)
    assert noise_types is not None and len(noise_types) > 0
    if noise_probs is None:
        idx = np.random.randint(len(noise_types))
        return noise_types[idx]
    noise_probs = np.asarray(_as_tuple(noise_probs), dtype=np.float64)
    assert len(noise_probs) == len(noise_types)
    noise_probs = noise_probs / noise_probs.sum()
    idx = np.random.choice(len(noise_types), p=noise_probs)
    return noise_types[idx]


def _sample_noise(noise_type: str, scale, shape):
    if noise_type == "laplace":
        return np.random.laplace(0, scale, size=shape)
    if noise_type == "gaussian":
        return np.random.normal(0, scale, size=shape)
    if noise_type == "uniform":
        half_width = np.sqrt(3.0) * scale
        return np.random.uniform(-half_width, half_width, size=shape)
    raise ValueError(f"unsupported noise_type: {noise_type}")


def _ensure_noisy_if_missing(asset: Asset, pc):
    if asset.sampled_vertices_noisy is None:
        asset.sampled_vertices_noisy = pc.copy()


def _normalize01(x: np.ndarray) -> np.ndarray:
    lo = np.percentile(x, 75.0)
    hi = np.percentile(x, 98.0)
    if hi <= lo + 1e-12:
        hi = float(np.max(x))
    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _patch_pca_risk_and_normal(patches: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    risks = []
    normals = []
    for patch in patches:
        n = patch.shape[0]
        kk = max(4, min(k, n))
        _, nn_idx = cKDTree(patch).query(patch, k=kk)
        neigh = patch[nn_idx]
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / max(kk - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        total = eigvals.sum(axis=1) + 1e-12
        curvatures = eigvals[:, 0] / total
        linearities = (eigvals[:, 2] - eigvals[:, 1]) / (eigvals[:, 2] + 1e-12)
        patch_normal = eigvecs[:, :, 0].astype(np.float32)
        patch_risk = np.maximum(
            _normalize01(curvatures),
            _normalize01(linearities),
        )
        risks.append(patch_risk[:, None])
        normals.append(patch_normal)
    return np.stack(risks, axis=0), np.stack(normals, axis=0)

@dataclass(frozen=True)
class Augment(ConfigSpec):
    
    @classmethod
    @abstractmethod
    def parse(cls, **kwags) -> 'Augment':
        pass
    
    @abstractmethod
    def apply(self, asset: Asset, **kwargs):
        pass

@dataclass(frozen=True)
class AugmentSample(Augment):
    
    num_samples: int # total number of vertices on the face to be sampled
    
    num_vertex_samples: int=0 # number of vertices to be chosen
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentSample':
        cls.check_keys(kwargs)
        return AugmentSample(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        assert asset.vertices is not None
        assert asset.faces is not None
        sampled_vertices, sampled_normals, sampled_vertex_groups, hidden_states = sample_vertex_groups(
            vertices=asset.vertices,
            faces=asset.faces,
            num_samples=self.num_samples,
            num_vertex_samples=self.num_vertex_samples,
        )
        asset.sampled_vertices = sampled_vertices

@dataclass(frozen=True)
class AugmentNormalizePC(Augment):
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentNormalizePC':
        cls.check_keys(kwargs)
        return AugmentNormalizePC(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentNormalizePC"
        p_max = pc.max(axis=0)
        p_min = pc.min(axis=0)
        center = (p_max + p_min) / 2
        pc = pc - center
        scale = np.sqrt((pc**2).sum(axis=1).max()).max()
        asset.sampled_vertices = pc / scale

@dataclass(frozen=True)
class AugmentAddNoise(Augment):
    
    noise_std_min: float
    
    noise_std_max: float

    noise_type: Union[str, Tuple[str, ...]]="laplace"

    noise_probs: Optional[Tuple[float, ...]]=None

    enabled: bool=True

    l2_noise_std: Optional[float]=None

    l2_noise_std_min: Optional[float]=None

    l2_noise_std_max: Optional[float]=None

    use_l2_as_noisy: bool=False
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddNoise':
        kwargs = deepcopy(kwargs)
        if "noise_type" in kwargs:
            kwargs["noise_type"] = _as_tuple(kwargs["noise_type"])
        if kwargs.get("noise_probs") is not None:
            kwargs["noise_probs"] = _as_tuple(kwargs["noise_probs"])
        cls.check_keys(kwargs)
        return AugmentAddNoise(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddNoise"
        if not self.enabled:
            _ensure_noisy_if_missing(asset, pc)
            return
        noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
        noise_type = _choose_noise_type(self.noise_type, self.noise_probs)
        noise = _sample_noise(noise_type, noise_std, pc.shape)
        asset.sampled_vertices_noisy = pc + noise
        l2_noise_std = self.l2_noise_std
        if self.l2_noise_std_min is not None and self.l2_noise_std_max is not None:
            l2_noise_std = np.random.uniform(self.l2_noise_std_min, self.l2_noise_std_max)
        if l2_noise_std is not None:
            noise_l2 = _sample_noise(noise_type, l2_noise_std, pc.shape)
            pc_noisy_l2 = pc + noise_l2
            if asset.meta is None:
                asset.meta = {}
            asset.meta['sampled_vertices_noisy_l2'] = pc_noisy_l2
            if self.use_l2_as_noisy:
                asset.sampled_vertices_noisy = pc_noisy_l2

@dataclass(frozen=True)
class AugmentAddMixedNoise(Augment):
    
    noise_std_min: float
    
    noise_std_max: float
    
    noise_types: Tuple[str, ...]=("laplace", "gaussian")
    
    noise_probs: Optional[Tuple[float, ...]]=None
    
    enabled: bool=True
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddMixedNoise':
        cls.check_keys(kwargs)
        kwargs = deepcopy(kwargs)
        if "noise_types" in kwargs:
            kwargs["noise_types"] = _as_tuple(kwargs["noise_types"])
        if kwargs.get("noise_probs") is not None:
            kwargs["noise_probs"] = _as_tuple(kwargs["noise_probs"])
        return AugmentAddMixedNoise(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddMixedNoise"
        if not self.enabled:
            _ensure_noisy_if_missing(asset, pc)
            return
        noise_type = _choose_noise_type(self.noise_types, self.noise_probs)
        noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
        noise = _sample_noise(noise_type, noise_std, pc.shape)
        asset.sampled_vertices_noisy = pc + noise

@dataclass(frozen=True)
class AugmentAddNonUniformNoise(Augment):
    
    noise_std_min: float
    
    noise_std_max: float
    
    noise_type: str="laplace"
    
    num_centers_min: int=1
    
    num_centers_max: int=4
    
    radius_min: float=0.08
    
    radius_max: float=0.25
    
    enabled: bool=True
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddNonUniformNoise':
        cls.check_keys(kwargs)
        return AugmentAddNonUniformNoise(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddNonUniformNoise"
        if not self.enabled:
            _ensure_noisy_if_missing(asset, pc)
            return
        
        N = pc.shape[0]
        num_centers = np.random.randint(self.num_centers_min, self.num_centers_max + 1)
        center_idx = np.random.choice(N, size=min(num_centers, N), replace=False)
        radius = np.random.uniform(self.radius_min, self.radius_max)
        
        weights = np.zeros((N,), dtype=np.float64)
        for idx in center_idx:
            dist2 = ((pc - pc[idx]) ** 2).sum(axis=1)
            weights = np.maximum(weights, np.exp(-dist2 / (2.0 * radius * radius + 1e-12)))
        
        scales = self.noise_std_min + (self.noise_std_max - self.noise_std_min) * weights
        noise = _sample_noise(self.noise_type, scales[:, None], pc.shape)
        asset.sampled_vertices_noisy = pc + noise

@dataclass(frozen=True)
class AugmentAddLocalStrongNoise(Augment):
    
    noise_std_min: float
    
    noise_std_max: float
    
    noise_type: str="laplace"
    
    num_centers_min: int=1
    
    num_centers_max: int=4
    
    radius_min: float=0.05
    
    radius_max: float=0.15
    
    enabled: bool=True
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddLocalStrongNoise':
        cls.check_keys(kwargs)
        return AugmentAddLocalStrongNoise(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddLocalStrongNoise"
        if not self.enabled:
            _ensure_noisy_if_missing(asset, pc)
            return
        
        noisy = asset.sampled_vertices_noisy.copy() if asset.sampled_vertices_noisy is not None else pc.copy()
        N = pc.shape[0]
        num_centers = np.random.randint(self.num_centers_min, self.num_centers_max + 1)
        center_idx = np.random.choice(N, size=min(num_centers, N), replace=False)
        tree = cKDTree(pc)
        
        mask = np.zeros((N,), dtype=bool)
        for idx in center_idx:
            radius = np.random.uniform(self.radius_min, self.radius_max)
            nn_idx = tree.query_ball_point(pc[idx], r=radius)
            mask[nn_idx] = True
        
        if mask.any():
            noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
            noise = _sample_noise(self.noise_type, noise_std, pc.shape)
            noisy[mask] = noisy[mask] + noise[mask]
        asset.sampled_vertices_noisy = noisy

@dataclass(frozen=True)
class AugmentLinear(Augment):
    
    scale: Tuple[float, float]=(1.0, 1.0)
    
    rotate_x_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_y_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_z_range: Tuple[float, float]=(0.0, 0.0)
    
    scale_p: float=0.0
    
    rotate_p: float=0.0
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentLinear':
        cls.check_keys(kwargs)
        return AugmentLinear(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        trans_vertex = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.rotate_p:
            r = random_euler_rotation(
                1,
                x_range=self.rotate_x_range,
                y_range=self.rotate_y_range,
                z_range=self.rotate_z_range,
            )[0]
            trans_vertex = r @ trans_vertex
        if np.random.rand() < self.scale_p:
            scale = np.zeros((4, 4), dtype=np.float32)
            scale[0, 0] = np.random.uniform(self.scale[0], self.scale[1])
            scale[1, 1] = np.random.uniform(self.scale[0], self.scale[1])
            scale[2, 2] = np.random.uniform(self.scale[0], self.scale[1])
            scale[3, 3] = 1.0
            trans_vertex = scale @ trans_vertex
        asset.transform(trans_vertex)
        rot = trans_vertex[:3, :3].transpose()
        trans = trans_vertex[:3, 3]
        if asset.sampled_vertices is not None:
            asset.sampled_vertices = np.matmul(asset.sampled_vertices, rot) + trans
        if asset.sampled_vertices_noisy is not None:
            asset.sampled_vertices_noisy = np.matmul(asset.sampled_vertices_noisy, rot) + trans
        if asset.meta is not None and 'sampled_vertices_noisy_l2' in asset.meta:
            asset.meta['sampled_vertices_noisy_l2'] = np.matmul(
                asset.meta['sampled_vertices_noisy_l2'],
                rot,
            ) + trans

@dataclass(frozen=True)
class AugmentPatch(Augment):
    
    patch_size: int
    
    num_patches: int
    
    train_cvm_network: bool

    straight_time: bool=False

    surface_target: bool=False

    use_noisy_l2: bool=False

    edge_risk: bool=False

    edge_risk_k: int=16

    edge_seed_candidate_multiplier: int=1

    edge_seed_prob: float=0.0

    edge_seed_score_percentile: float=90.0
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentPatch':
        cls.check_keys(kwargs)
        return AugmentPatch(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        pc_noisy = asset.sampled_vertices_noisy
        
        assert pc is not None
        assert pc_noisy is not None

        if self.use_noisy_l2 and asset.meta is not None and 'sampled_vertices_noisy_l2' in asset.meta:
            pc_noisy = asset.meta['sampled_vertices_noisy_l2']
        
        N = pc_noisy.shape[0]
        
        num_candidates = self.num_patches
        if (
            self.edge_risk and
            self.edge_seed_candidate_multiplier > 1 and
            np.random.rand() < self.edge_seed_prob
        ):
            num_candidates = min(N, self.num_patches * self.edge_seed_candidate_multiplier)

        seed_idx = np.random.permutation(N)[:num_candidates]     # (P,)
        seed_points = pc_noisy[seed_idx]                         # (P, 3)
        
        tree = cKDTree(pc_noisy)
        _, nn_idx = tree.query(seed_points, k=self.patch_size)   # (P, M)

        pat_A = pc_noisy[nn_idx]  # (P, M, 3)
        pat_clean_corr = pc[nn_idx]  # original clean counterpart for each noisy point
        if self.surface_target:
            clean_tree = cKDTree(pc)
            _, surface_idx = clean_tree.query(pat_A.reshape(-1, 3), k=1)
            pat_B = pc[surface_idx].reshape(num_candidates, self.patch_size, 3)
            _, seed_surface_idx = clean_tree.query(seed_points, k=1)
            seed_targets = pc[seed_surface_idx]
        else:
            pat_B = pc[nn_idx]        # (P, M, 3)
            seed_targets = pc[seed_idx]

        l1, l2 = 1e-8, 1.0
        if self.straight_time:
            t = np.random.rand(num_candidates, 1, 1)
            t = np.broadcast_to(t, (num_candidates, self.patch_size, 1))
        else:
            t = np.random.rand(num_candidates, self.patch_size, 1)
        t = (l2 - l1) * t + l1
        
        pat_t = t * pat_B + (1 - t) * pat_A
        seed_points_t = (
            t[:, 0:1, :] * seed_targets[:, None, :] +
            (1 - t[:, 0:1, :]) * pc_noisy[seed_idx][:, None, :]
        )
        
        pat_A = pat_A - seed_points_t
        pat_B = pat_B - seed_points_t
        pat_t = pat_t - seed_points_t
        pat_clean_corr = pat_clean_corr - seed_points_t
        if self.edge_risk:
            pc_edge_risk, pc_normal = _patch_pca_risk_and_normal(
                patches=pat_B,
                k=self.edge_risk_k,
            )
            if num_candidates > self.num_patches:
                patch_score = np.percentile(
                    pc_edge_risk[:, :, 0],
                    self.edge_seed_score_percentile,
                    axis=1,
                )
                select_idx = np.argsort(patch_score)[-self.num_patches:]
                select_idx = np.sort(select_idx)
                pat_A = pat_A[select_idx]
                pat_B = pat_B[select_idx]
                pat_t = pat_t[select_idx]
                pat_clean_corr = pat_clean_corr[select_idx]
                pc_edge_risk = pc_edge_risk[select_idx]
                pc_normal = pc_normal[select_idx]
                t = t[select_idx]
        
        if asset.meta is None:
            asset.meta = {}
        asset.meta['pc_noisy'] = pat_A
        asset.meta['pc_clean'] = pat_B
        asset.meta['pc_clean_corr'] = pat_clean_corr
        asset.meta['pc_mix'] = pat_t
        if self.edge_risk:
            asset.meta['pc_edge_risk'] = pc_edge_risk
            asset.meta['pc_normal'] = pc_normal
        if self.straight_time:
            asset.meta['pc_time'] = t[:, 0, 0]

def get_augments(*args) -> List[Augment]:
    MAP = {
        "sample": AugmentSample,
        "normalize_pc": AugmentNormalizePC,
        "add_noise": AugmentAddNoise,
        "mixed_noise": AugmentAddMixedNoise,
        "nonuniform_noise": AugmentAddNonUniformNoise,
        "local_strong_noise": AugmentAddLocalStrongNoise,
        "linear": AugmentLinear,
        "patch": AugmentPatch,
    }
    MAP: Dict[str, type[Augment]]
    augments = []
    for (i, config) in enumerate(args):
        __target__ = config.get('__target__')
        assert __target__ is not None, f"do not find `__target__` in augment of position {i}"
        c = deepcopy(config)
        del c['__target__']
        augments.append(MAP[__target__].parse(**c))
    return augments
