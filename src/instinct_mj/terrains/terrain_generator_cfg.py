from dataclasses import dataclass
from typing import Literal

from mjlab.terrains import FlatPatchSamplingCfg as FlatPatchSamplingCfgBase
from mjlab.terrains import TerrainGeneratorCfg as TerrainGeneratorCfgBase

from .terrain_generator import FiledTerrainGenerator


@dataclass
class MultiRadiusFlatPatchSamplingCfg(FlatPatchSamplingCfgBase):
    """Flat-patch sampling config that accepts multiple footprint radii."""

    patch_radius: float | list[float] | tuple[float, ...] = 0.5


@dataclass(kw_only=True)
class FiledTerrainGeneratorCfg(TerrainGeneratorCfgBase):
    class_type: type = FiledTerrainGenerator
    horizontal_scale: float | None = None
    """Optional generator-wide horizontal scale propagated to sub-terrains that define it."""
    vertical_scale: float | None = None
    """Optional generator-wide vertical scale propagated to sub-terrains that define it."""
    slope_threshold: float | None = None
    """Optional generator-wide slope threshold propagated to sub-terrains that define it."""
    hfield_resolution: float | None = None
    """Sampling resolution (meters) for filed-generator mesh->hfield conversion.

    If None, fallback to sub-terrain ``horizontal_scale`` when available.
    """

    hfield_base_thickness_ratio: float = 1.0
    """Base thickness ratio used for generated collision hfield."""

    hfield_num_workers: int = 0
    """Number of workers for filed-generator mesh->hfield ray-casting conversion (0 means auto)."""

    hfield_raycast_backend: Literal["cpu", "gpu"] = "cpu"
    """Ray-cast backend for mesh->hfield conversion.

    - ``"cpu"``: Use trimesh ray-casting (supports multi-process workers).
    - ``"gpu"``: Use Warp GPU ray-casting on ``hfield_gpu_device``.
    """

    hfield_gpu_device: str = "cuda"
    """Torch/Warp device used when ``hfield_raycast_backend='gpu'``."""

    hfield_gpu_batch_size: int = 262144
    """Ray batch size for GPU ray-casting. Increase for throughput, decrease for lower memory."""

    hfield_stitch_border_width: float = 0.0
    """Extra stitched flat border width (meters) applied to every sub-terrain edge.

    This is useful when different terrain types should share a consistently flat
    outer ring so neighboring tiles are height-aligned at seams.
    """
