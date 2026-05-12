from __future__ import annotations

from dataclasses import dataclass, field

from .constants import um


@dataclass(frozen=True)
class ModalStructureParams:
    lambda_0: float = 1.0 * um
    n_points: int = 1000
    refractive_indices: tuple[float, ...] = (3.55, 3.55, 3.60, 3.55, 3.55)
    thicknesses: tuple[float, ...] = (
        5.0 * um,
        0.5 * um,
        1.0 * um,
        0.5 * um,
        5.0 * um,
    )

    def __post_init__(self) -> None:
        if len(self.refractive_indices) != len(self.thicknesses):
            raise ValueError("refractive_indices e thicknesses devem ter o mesmo tamanho.")
        if self.n_points < 3:
            raise ValueError("n_points deve ser >= 3.")

    @property
    def n_core(self) -> float:
        return max(self.refractive_indices)

    @property
    def n_clad(self) -> float:
        return min(self.refractive_indices[0], self.refractive_indices[-1])


@dataclass(frozen=True)
class CouplerTestParams:
    n_core: float = 1.50
    n_clad: float = 1.42
    window: float = 40.0 * um
    lambda_launch: float = 1.0 * um
    length_total: float = 1764.0 * um
    length_coupling: float = 882.0 * um
    port_offset: float = 10.0 * um
    core_width: float = 1.0 * um
    center_offset: float = 1.0 * um
    sigma: float = 0.45 * um
    n_points: int = 401
    mode_search_count: int = 8

    def __post_init__(self) -> None:
        if self.n_points < 3:
            raise ValueError("n_points deve ser >= 3.")
        if self.length_total <= self.length_coupling:
            raise ValueError("length_total deve ser maior que length_coupling.")

    @property
    def bend_length(self) -> float:
        return 0.5 * (self.length_total - self.length_coupling)
