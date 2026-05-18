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

    noise_type: str="laplace"

    enabled: bool=True
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddNoise':
        cls.check_keys(kwargs)
        return AugmentAddNoise(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddNoise"
        if not self.enabled:
            _ensure_noisy_if_missing(asset, pc)
            return
        noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
        noise = _sample_noise(self.noise_type, noise_std, pc.shape)
        asset.sampled_vertices_noisy = pc + noise

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

@dataclass(frozen=True)
class AugmentPatch(Augment):
    
    patch_size: int
    
    num_patches: int
    
    train_cvm_network: bool
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentPatch':
        cls.check_keys(kwargs)
        return AugmentPatch(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        pc_noisy = asset.sampled_vertices_noisy
        
        assert pc is not None
        assert pc_noisy is not None
        
        N = pc_noisy.shape[0]
        
        seed_idx = np.random.permutation(N)[:self.num_patches]   # (P,)
        seed_points = pc_noisy[seed_idx]                         # (P, 3)
        
        tree = cKDTree(pc_noisy)
        _, nn_idx = tree.query(seed_points, k=self.patch_size)   # (P, M)

        pat_A = pc_noisy[nn_idx]  # (P, M, 3)
        pat_B = pc[nn_idx]        # (P, M, 3)

        l1, l2 = 1e-8, 1.0
        t = np.random.rand(self.num_patches, self.patch_size, 1)
        t = (l2 - l1) * t + l1
        
        pat_t = t * pat_B + (1 - t) * pat_A
        seed_points_t = (
            t[:, 0:1, :] * pc[seed_idx][:, None, :] +
            (1 - t[:, 0:1, :]) * pc_noisy[seed_idx][:, None, :]
        )
        
        pat_A = pat_A - seed_points_t
        pat_B = pat_B - seed_points_t
        pat_t = pat_t - seed_points_t
        
        if asset.meta is None:
            asset.meta = {}
        asset.meta['pc_noisy'] = pat_A
        asset.meta['pc_clean'] = pat_B
        asset.meta['pc_mix'] = pat_t

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
