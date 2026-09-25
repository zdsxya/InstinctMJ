"""Explicit, positive-parameter engineering profile for MILD contact.

The paper's ambiguous signed coefficients are not silently reproduced. See
docs/mild-contact.md for the positive-depth convention and discretization.
"""

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class MildMaterialCfg:
    alpha: tuple[float, float] = (0.8, 1.2)
    recruitment: tuple[float, float] = (0.8, 1.8)
    mass_scale: tuple[float, float] = (0.5, 2.0)
    drag_scale: tuple[float, float] = (1.0, 6.0)
    volume_fraction: tuple[float, float] = (0.50, 0.64)
    density: tuple[float, float] = (1600.0, 1600.0)
    friction: tuple[float, float] = (0.4, 1.0)
    penetration_stiffness: tuple[float, float] = (20.0, 200.0)
    conical_stress: tuple[float, float] = (1.0e4, 1.2e5)
    drift_stiffness: tuple[float, float] = (1000.0, 4000.0)
    drift_damping: tuple[float, float] = (0.5, 2.0)


@dataclass(kw_only=True)
class MildContactCfg:
    material: MildMaterialCfg = field(default_factory=MildMaterialCfg)
    randomize_material: bool = True
    cell_size: float = 0.0195
    quadrature_subdivisions: int = 16
    reference_cell_size: float = 0.0195
    surface_height: float = 0.0
    max_penetration: float = 0.20
    friction_iterations: int = 24
    coupling_iterations: int = 3
    impulse_tolerance: float = 1.0e-4
    contact_threshold: float = 0.1
    debug_vis: bool = False
    use_cuda_graph: bool = True

    def validate(self) -> None:
        import math
        from dataclasses import fields

        if self.cell_size <= 0 or self.reference_cell_size <= 0:
            raise ValueError("MILD cell sizes must be positive")
        if self.quadrature_subdivisions < 1 or self.max_penetration <= 0:
            raise ValueError("MILD quadrature and maximum penetration must be positive")
        if self.friction_iterations < 1 or self.coupling_iterations < 1:
            raise ValueError("MILD solver iteration counts must be positive")
        if self.impulse_tolerance <= 0 or self.contact_threshold < 0:
            raise ValueError("Invalid MILD solver/contact tolerance")
        for item in fields(self.material):
            lo, hi = getattr(self.material, item.name)
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi):
                raise ValueError(f"Invalid positive-depth MILD range: {item.name}")
        if not 0 < self.material.alpha[0] <= self.material.alpha[1] < math.pi / 2:
            raise ValueError("MILD shear angle must lie strictly between 0 and pi/2")
        if not 0 < self.material.volume_fraction[0] <= self.material.volume_fraction[1] <= 1:
            raise ValueError("MILD volume fraction must lie in (0, 1]")
        if self.material.density[0] <= 0 or self.material.penetration_stiffness[0] <= 0:
            raise ValueError("MILD density and penetration stiffness must be positive")
