from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

from .bpm_tridiagonal import (
    crank_nicolson_tridiagonal_coefficients,
    pack_tridiagonal_banded,
    solve_banded_tridiagonal,
    tridiagonal_matvec,
)
from .modal import assemble_helmholtz_operator, solve_guided_modes
from .parameters import CouplerTestParams

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration
    njit = None


@dataclass(frozen=True)
class CouplerSimulationResult:
    lambda_launch: float
    x_vec: np.ndarray
    z_vec: np.ndarray
    offset_trace: np.ndarray
    n_profile_center: np.ndarray
    n_effs_local: np.ndarray
    modes_local: np.ndarray
    n_ref_local: float
    dz_local: float
    theta_spec_local: float
    theta_geom_local: float
    k_trans_local: float
    e_input: np.ndarray
    e_evol: np.ndarray
    power_total: np.ndarray
    power_upper: np.ndarray
    power_lower: np.ndarray


if njit is not None:

    # cache=False evita RuntimeError em alguns filesystems montados
    # (ex.: paths do Windows via /mnt/c no WSL) sem "locator" de cache.
    @njit(cache=False)
    def _compute_powers_numba(e_evol, x_vec, offset_trace, core_width):
        nz, nx = e_evol.shape
        power_total = np.zeros(nz)
        power_upper = np.zeros(nz)
        power_lower = np.zeros(nz)

        for iz in range(nz):
            offset_now = offset_trace[iz]
            total = 0.0
            upper = 0.0
            lower = 0.0
            for ix in range(nx - 1):
                x0 = x_vec[ix]
                x1 = x_vec[ix + 1]
                dx = x1 - x0

                y0 = e_evol[iz, ix].real * e_evol[iz, ix].real + e_evol[iz, ix].imag * e_evol[iz, ix].imag
                y1 = (
                    e_evol[iz, ix + 1].real * e_evol[iz, ix + 1].real
                    + e_evol[iz, ix + 1].imag * e_evol[iz, ix + 1].imag
                )
                avg = 0.5 * (y0 + y1) * dx
                total += avg

                upper0 = abs(x0 - offset_now) <= (core_width / 2)
                upper1 = abs(x1 - offset_now) <= (core_width / 2)
                if upper0 and upper1:
                    upper += avg

                lower0 = abs(x0 + offset_now) <= (core_width / 2)
                lower1 = abs(x1 + offset_now) <= (core_width / 2)
                if lower0 and lower1:
                    lower += avg

            power_total[iz] = total
            power_upper[iz] = upper
            power_lower[iz] = lower

        return power_total, power_upper, power_lower


def _compute_powers_numpy(e_evol, x_vec, offset_trace, core_width):
    power_total = np.zeros(len(offset_trace))
    power_upper = np.zeros(len(offset_trace))
    power_lower = np.zeros(len(offset_trace))

    for iz, offset_now in enumerate(offset_trace):
        intensity_now = np.abs(e_evol[iz, :]) ** 2
        upper_mask = np.abs(x_vec - offset_now) <= (core_width / 2)
        lower_mask = np.abs(x_vec + offset_now) <= (core_width / 2)
        power_total[iz] = trapezoid(intensity_now, x_vec)
        power_upper[iz] = trapezoid(intensity_now[upper_mask], x_vec[upper_mask])
        power_lower[iz] = trapezoid(intensity_now[lower_mask], x_vec[lower_mask])

    return power_total, power_upper, power_lower


def generate_gaussian_input(x_vec: np.ndarray, xc: float, sigma: float) -> np.ndarray:
    return np.exp(-((x_vec - xc) ** 2) / (2.0 * sigma**2))


def normalize_mode(x_vec: np.ndarray, field: np.ndarray, n_profile: np.ndarray | None = None) -> np.ndarray:
    norm = np.sqrt(trapezoid(np.abs(field) ** 2, x_vec))
    if norm == 0:
        raise ValueError("Campo com norma nula nao pode ser normalizado.")
    return field / norm


def coupler_offset(z_pos: float, params: CouplerTestParams) -> float:
    if z_pos <= params.bend_length:
        return params.port_offset + (params.center_offset - params.port_offset) * (
            z_pos / params.bend_length
        )
    if z_pos <= params.bend_length + params.length_coupling:
        return params.center_offset

    z_local = z_pos - (params.bend_length + params.length_coupling)
    return params.center_offset + (params.port_offset - params.center_offset) * (
        z_local / params.bend_length
    )


def build_coupler_profile(x_vec: np.ndarray, offset: float, params: CouplerTestParams) -> np.ndarray:
    n_prof = np.full_like(x_vec, params.n_clad, dtype=float)
    upper_core = np.abs(x_vec - offset) <= (params.core_width / 2)
    lower_core = np.abs(x_vec + offset) <= (params.core_width / 2)
    n_prof[upper_core | lower_core] = params.n_core
    return n_prof


def run_bpm_coupler(
    initial_field: np.ndarray,
    x_vec: np.ndarray,
    z_vec: np.ndarray,
    n_ref: float,
    k0_local: float,
    params: CouplerTestParams,
) -> np.ndarray:
    nx_local = len(x_vec)
    dx_local = x_vec[1] - x_vec[0]
    dz_local = z_vec[1] - z_vec[0]
    alpha1_local = -1j / (2 * k0_local * n_ref)
    alpha2_local = -1j * k0_local / (2 * n_ref)
    t1_local = 1.0 / dx_local**2
    field_history = np.zeros((len(z_vec), nx_local), dtype=complex)
    field_history[0, :] = initial_field
    operator_cache = {}

    for iz in range(1, len(z_vec)):
        z_mid = 0.5 * (z_vec[iz] + z_vec[iz - 1])
        offset_mid = coupler_offset(z_mid, params)
        edge_key = tuple(
            np.searchsorted(
                x_vec,
                np.array(
                    [
                        -offset_mid - params.core_width / 2,
                        -offset_mid + params.core_width / 2,
                        offset_mid - params.core_width / 2,
                        offset_mid + params.core_width / 2,
                    ]
                ),
            )
        )

        if edge_key not in operator_cache:
            n_prof_mid = build_coupler_profile(x_vec, offset_mid, params)
            lo_a, mid_a, up_a, lo_b, mid_b, up_b = crank_nicolson_tridiagonal_coefficients(
                alpha1_local,
                alpha2_local,
                t1_local,
                dz_local,
                n_prof_mid,
                n_ref,
            )
            ab_a = pack_tridiagonal_banded(lo_a, mid_a, up_a)
            operator_cache[edge_key] = (ab_a, lo_b, mid_b, up_b)

        ab_a, lo_b, mid_b, up_b = operator_cache[edge_key]
        rhs = tridiagonal_matvec(lo_b, mid_b, up_b, field_history[iz - 1, :])
        field_history[iz, :] = solve_banded_tridiagonal(ab_a, rhs)

    return field_history


def simulate_coupler(
    params: CouplerTestParams | None = None,
    lambda_launch: float | None = None,
) -> CouplerSimulationResult:
    params = params or CouplerTestParams()
    lambda_launch = params.lambda_launch if lambda_launch is None else lambda_launch
    k0_local = 2 * np.pi / lambda_launch
    x_vec = np.linspace(-params.window / 2, params.window / 2, params.n_points)
    dx_local = x_vec[1] - x_vec[0]
    theta_spec_local = np.arctan((10.0e-6) / params.bend_length)
    theta_geom_local = np.arctan((params.port_offset - params.center_offset) / params.bend_length)

    n_profile_center = build_coupler_profile(x_vec, params.center_offset, params)
    a_local, b_local = assemble_helmholtz_operator(n_profile_center, dx_local, k0_local)
    n_effs_local, modes_local = solve_guided_modes(
        a_local,
        b_local,
        params.n_core,
        params.n_clad,
        k_search=params.mode_search_count,
    )
    if len(n_effs_local):
        n_ref_local = float(n_effs_local[0])
    else:
        n_ref_local = 0.5 * (params.n_core + params.n_clad)

    dz_local = lambda_launch / (4 * n_ref_local)
    z_vec = np.arange(0.0, params.length_total + 0.5 * dz_local, dz_local)
    offset_trace = np.array([coupler_offset(z_pos, params) for z_pos in z_vec])
    k_trans_local = k0_local * params.n_core * np.sin(theta_spec_local)
    e_gauss = generate_gaussian_input(x_vec, xc=params.port_offset, sigma=params.sigma)
    e_input = normalize_mode(
        x_vec,
        e_gauss * np.exp(-1j * k_trans_local * (x_vec - params.port_offset)),
        n_profile_center,
    )
    e_evol = run_bpm_coupler(e_input, x_vec, z_vec, n_ref_local, k0_local, params)
    if njit is not None:
        power_total, power_upper, power_lower = _compute_powers_numba(
            e_evol, x_vec, offset_trace, params.core_width
        )
    else:
        power_total, power_upper, power_lower = _compute_powers_numpy(
            e_evol, x_vec, offset_trace, params.core_width
        )

    return CouplerSimulationResult(
        lambda_launch=lambda_launch,
        x_vec=x_vec,
        z_vec=z_vec,
        offset_trace=offset_trace,
        n_profile_center=n_profile_center,
        n_effs_local=n_effs_local,
        modes_local=modes_local,
        n_ref_local=n_ref_local,
        dz_local=dz_local,
        theta_spec_local=theta_spec_local,
        theta_geom_local=theta_geom_local,
        k_trans_local=k_trans_local,
        e_input=e_input,
        e_evol=e_evol,
        power_total=power_total,
        power_upper=power_upper,
        power_lower=power_lower,
    )
