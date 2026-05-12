from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.sparse import diags

from .analyzer import WaveguideModalBPM1D
from .bpm_tridiagonal import (
    crank_nicolson_tridiagonal_coefficients,
    pack_tridiagonal_banded,
    solve_banded_tridiagonal,
    tridiagonal_matvec,
)
from .constants import nm, um
from .coupling_sweep import (
    estimate_first_relevant_peak,
    estimate_lc_by_supermodes,
    plot_parameter_lc,
    plot_parameter_peak_transfer,
    plot_parameter_to_lc,
)
from .parameters import CouplerTestParams
from .test_cases import CouplerSimulationResult, simulate_coupler

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - optional acceleration
    Parallel = None
    delayed = None


@dataclass(frozen=True)
class StraightBPMResult:
    x: np.ndarray
    z: np.ndarray
    e_evol: np.ndarray
    n_profile: np.ndarray
    mode_reference: np.ndarray
    core_thickness: float
    length_total: float


@dataclass(frozen=True)
class CouplerT3Result:
    x: np.ndarray
    z: np.ndarray
    n_profile: np.ndarray
    n_effs: np.ndarray
    modes: np.ndarray
    e_evol: np.ndarray
    power_upper: np.ndarray
    power_lower: np.ndarray
    power_upper_norm: np.ndarray
    power_lower_norm: np.ndarray
    lc_idx: int
    lc: float
    gap: float
    core_width: float
    guide_center: float
    lambda_launch: float
    length_total: float
    sigma: float


def print_mode_summary(n_effs: Sequence[float], *, label: str = "Part 1 success") -> None:
    n_effs_arr = np.asarray(n_effs, dtype=float)
    if len(n_effs_arr) == 0:
        print("No guided modes found.")
        return
    print(f"{label}: {len(n_effs_arr)} guided mode(s) found.")
    for i, n_eff in enumerate(n_effs_arr):
        print(f"Mode {i}: neff = {n_eff:.5f}")


def plot_multilayer_modes(
    *,
    x: np.ndarray,
    n_profile: np.ndarray,
    boundaries: np.ndarray,
    refractive_indices: Sequence[float],
    thicknesses: Sequence[float],
    lambda_0: float,
    modes: np.ndarray,
    n_effs: Sequence[float],
    polarization: str = "TE",
    field_symbol: str = "E",
):
    n_effs_arr = np.asarray(n_effs, dtype=float)
    if len(n_effs_arr) == 0:
        raise ValueError("No guided mode was found to plot.")

    fig, axes = plt.subplots(
        nrows=len(n_effs_arr) + 1,
        ncols=1,
        figsize=(10, 2 * (len(n_effs_arr) + 2)),
        sharex=True,
    )

    axes[0].plot(x / um, n_profile, "k-", lw=1, alpha=0.8)
    axes[0].set_ylabel(r"Index $n$")
    axes[0].set_title(
        fr"Multilayer Profile and {polarization.upper()} Modes ($\lambda$ = {lambda_0 / nm:.0f} nm)"
    )

    cmap = plt.colormaps.get_cmap("Pastel1")
    color_idx = np.linspace(0, 1, len(refractive_indices))
    x_ini = -float(np.sum(thicknesses)) / 2.0

    for i, (n_i, thickness) in enumerate(zip(refractive_indices, thicknesses)):
        x_fim = x_ini + thickness
        axes[0].fill_between(
            [x_ini / um, x_fim / um],
            [min(refractive_indices) - 0.01] * 2,
            [max(refractive_indices) + 0.01] * 2,
            color=cmap(color_idx[i]),
            alpha=0.3,
            label=f"Layer {i} (n={n_i:.2f})",
        )
        x_ini = x_fim

    axes[0].legend(loc="upper right", fontsize="x-small", ncol=2)
    for b in boundaries[:-1]:
        axes[0].axvline(b / um, color="gray", lw=0.5, ls=":")

    for i in range(len(n_effs_arr)):
        ax = axes[i + 1]
        norm_field = modes[:, i] / np.max(np.abs(modes[:, i]))
        ax.plot(
            x / um,
            norm_field,
            label=fr"{polarization.upper()}$_{{{i}}}$ ($n_{{eff}}$={n_effs_arr[i]:.4f})",
        )
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        for b in boundaries[:-1]:
            ax.axvline(b / um, color="red", ls=":", alpha=0.3)
        ax.set_ylabel(fr"${field_symbol}_{{{i}}}(x)$")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("x [um]")
    fig.tight_layout()
    return fig, axes


def plot_dispersion_with_angle_axis(
    *,
    d_norm_vec: np.ndarray,
    neff_matrix: np.ndarray,
    max_modes: int,
    n_clad_edge: float,
    n_core: float,
    lambda_0: float,
):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title(fr"TE/TM Modal Dispersion Curves ($\lambda$ = {lambda_0 / nm:.0f} nm)")

    for m in range(max_modes):
        mask = ~np.isnan(neff_matrix[m, :])
        d_plot = d_norm_vec[mask]
        n_plot = neff_matrix[m, mask]
        if len(d_plot) == 0:
            continue
        line, = ax1.plot(d_plot, n_plot, lw=1)
        if len(d_plot) > 10:
            idx_txt = len(d_plot) // 3
            ax1.text(
                d_plot[idx_txt] - 0.1,
                n_plot[idx_txt] + 0.001,
                fr"TE$_{{{m}}}$, TM$_{{{m}}}$",
                fontsize=10,
                rotation=35,
                ha="center",
                color=line.get_color(),
            )

    ax1.set_xlabel(r"$d / \lambda$ (core thickness)")
    ax1.set_xlim(0, max(d_norm_vec))
    ax1.set_ylabel(r"Effective index $n_{eff}$")
    ax1.set_ylim(n_clad_edge, n_core)

    y_ticks = np.linspace(n_clad_edge, n_core, 6)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([f"{v:.4g}" for v in y_ticks])

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Propagation angle $\theta$ (degrees)", rotation=-90, labelpad=25)
    ax2.set_ylim(n_clad_edge, n_core)

    theta_c_deg = np.degrees(np.arcsin(n_clad_edge / n_core))
    angle_step = max(1, int(np.round((90 - theta_c_deg) / 8)))
    mid_angles = list(np.arange(int(np.ceil(theta_c_deg)), 90, angle_step))
    if mid_angles and (90.0 - mid_angles[-1]) < 2.0:
        mid_angles.pop()

    target_angles = [theta_c_deg] + mid_angles + [90.0]
    ax2.set_yticks(n_core * np.sin(np.radians(target_angles)))

    angle_labels = []
    for a in target_angles:
        if a == theta_c_deg:
            angle_labels.append(fr"{a:.1f}° = $\theta_c$ (cutoff)")
        elif a == 90.0:
            angle_labels.append("90° (axial ray)")
        else:
            angle_labels.append(f"{int(a)}°")
    ax2.set_yticklabels(angle_labels)

    fig.tight_layout()
    return fig, (ax1, ax2)


def _core_thickness_from_layers(thicknesses: Sequence[float], refractive_indices: Sequence[float]) -> float:
    core_idx = int(np.argmax(np.asarray(refractive_indices, dtype=float)))
    return float(thicknesses[core_idx])


def run_straight_waveguide_bpm_fundamental(
    analyzer: WaveguideModalBPM1D,
    modal_te: dict[str, Any],
    *,
    length_total: float,
    refractive_indices: Sequence[float],
    thicknesses: Sequence[float],
) -> StraightBPMResult:
    bpm_context = analyzer.create_bpm_context(modal_te, length_total=length_total)
    x = modal_te["x"]
    n_profile = modal_te["n_profile"]
    modes = modal_te["modes"]
    n_effs = modal_te["n_effs"]
    n_points = len(x)

    e_evol = analyzer.run_bpm_propagation(
        modes[:, 0],
        bpm_context["nz"],
        n_points,
        bpm_context["alpha1"],
        bpm_context["alpha2"],
        bpm_context["T1"],
        bpm_context["laplacian"],
        bpm_context["I"],
        bpm_context["dz"],
        n_profile,
        n_effs,
    )
    z = np.arange(bpm_context["nz"]) * bpm_context["dz"]
    return StraightBPMResult(
        x=x,
        z=z,
        e_evol=e_evol,
        n_profile=n_profile,
        mode_reference=modes[:, 0],
        core_thickness=_core_thickness_from_layers(thicknesses, refractive_indices),
        length_total=length_total,
    )


def run_straight_waveguide_bpm_gaussian(
    analyzer: WaveguideModalBPM1D,
    modal_te: dict[str, Any],
    *,
    length_total: float,
    refractive_indices: Sequence[float],
    thicknesses: Sequence[float],
    xc_input: float = 0.0,
    sigma: float | None = None,
) -> StraightBPMResult:
    core_thickness = _core_thickness_from_layers(thicknesses, refractive_indices)
    sigma_now = 0.5 * core_thickness if sigma is None else sigma

    bpm_context = analyzer.create_bpm_context(modal_te, length_total=length_total)
    x = modal_te["x"]
    n_profile = modal_te["n_profile"]
    modes = modal_te["modes"]
    n_effs = modal_te["n_effs"]
    n_points = len(x)

    e_gauss = analyzer.generate_gaussian_input(x, xc=xc_input, sigma=sigma_now)
    initial_field = analyzer.normalize_field(x, e_gauss)

    e_evol = analyzer.run_bpm_propagation(
        initial_field,
        bpm_context["nz"],
        n_points,
        bpm_context["alpha1"],
        bpm_context["alpha2"],
        bpm_context["T1"],
        bpm_context["laplacian"],
        bpm_context["I"],
        bpm_context["dz"],
        n_profile,
        n_effs,
    )
    z = np.arange(bpm_context["nz"]) * bpm_context["dz"]
    return StraightBPMResult(
        x=x,
        z=z,
        e_evol=e_evol,
        n_profile=n_profile,
        mode_reference=modes[:, 0],
        core_thickness=core_thickness,
        length_total=length_total,
    )


def plot_straight_bpm_heatmap(
    result: StraightBPMResult,
    *,
    title: str,
    cmap: str = "inferno",
    vmax_scale: float = 1.0,
    xlim_factor: float = 5.0,
):
    intensity = np.abs(result.e_evol) ** 2
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        intensity,
        extent=[result.x[0] / um, result.x[-1] / um, result.length_total / um, 0],
        aspect="auto",
        cmap=cmap,
        vmax=np.max(intensity) * vmax_scale,
    )
    fig.colorbar(im, label=r"Intensidade Normalizada $|E|^2$")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(r"Transverse Position $x$ [um]")
    ax.set_ylabel(r"Propagation Distance $z$ [um]")

    x_half = xlim_factor * result.core_thickness / um
    ax.set_xlim([-x_half, x_half])
    ax.axvline(-result.core_thickness / (2 * um), color="white", ls=":", alpha=0.5)
    ax.axvline(result.core_thickness / (2 * um), color="white", ls=":", alpha=0.5)
    fig.tight_layout()
    return fig, ax


def plot_straight_profile_comparison(
    result: StraightBPMResult,
    *,
    title: str,
    xlim_factor: float = 2.0,
):
    input_profile = np.abs(result.e_evol[0, :])
    output_profile = np.abs(result.e_evol[-1, :])
    theoretical_mode = np.abs(result.mode_reference)

    input_profile /= np.max(input_profile)
    output_profile /= np.max(output_profile)
    theoretical_mode /= np.max(theoretical_mode)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.x / um, input_profile, "r--", lw=4, alpha=0.5, label="Input (z=0)")
    ax.plot(result.x / um, output_profile, "b-", lw=1.5, alpha=0.9, label="Output (BPM z=L)")
    ax.plot(result.x / um, theoretical_mode, "k:", lw=2, label="Theoretical Mode")
    ax.axvspan(
        -result.core_thickness / (2 * um),
        result.core_thickness / (2 * um),
        color="yellow",
        alpha=0.1,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Transverse Position x [um]")
    ax.set_ylabel("Normalized Amplitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    x_half = xlim_factor * result.core_thickness / um
    ax.set_xlim([-x_half, x_half])
    return fig, ax


def run_parallel_coupler_bpm(
    analyzer: WaveguideModalBPM1D,
    *,
    x: np.ndarray,
    dx: float,
    refractive_indices: Sequence[float],
    thicknesses: Sequence[float],
    gap: float = 1.0 * um,
    lambda_launch: float = 1550.0 * nm,
    length_total: float = 2000.0 * um,
    sigma_ratio: float = 0.45,
    k_search: int = 6,
) -> CouplerT3Result:
    core_width = _core_thickness_from_layers(thicknesses, refractive_indices)
    guide_center = 0.5 * (core_width + gap)
    sigma = sigma_ratio * core_width
    k0_local = 2 * np.pi / lambda_launch

    n_profile = analyzer.build_dual_core_profile(
        x_vec=x,
        center_offset=guide_center,
        core_width=core_width,
        n_core=max(refractive_indices),
        n_clad=refractive_indices[0],
    )
    a_t3, b_t3 = analyzer.assemble_helmholtz_operator(n_profile, dx, k0_local)
    n_effs, modes = analyzer.solve_guided_modes(
        a_t3,
        b_t3,
        n_core=max(refractive_indices),
        n_cladding=refractive_indices[0],
        k_search=k_search,
    )

    dz = lambda_launch / (4 * n_effs[0])
    nz = int(length_total / dz)
    alpha1 = -1j / (2 * k0_local * n_effs[0])
    alpha2 = -1j * k0_local / (2 * n_effs[0])
    n_points = len(x)

    identity = diags([np.ones(n_points)], [0], format="csr")
    laplacian = diags(
        [np.ones(n_points - 1), np.full(n_points, -2.0), np.ones(n_points - 1)],
        [-1, 0, 1],
        format="csr",
    )
    t1 = 1.0 / dx**2

    e_launch = analyzer.generate_gaussian_input(x, xc=guide_center, sigma=sigma)
    initial_field = analyzer.normalize_field(x, e_launch)
    e_evol = analyzer.run_bpm_propagation(
        initial_field,
        nz,
        n_points,
        alpha1,
        alpha2,
        t1,
        laplacian,
        identity,
        dz,
        n_profile,
        n_effs,
    )
    z = np.arange(nz) * dz

    upper_mask = np.abs(x - guide_center) <= (core_width / 2)
    lower_mask = np.abs(x + guide_center) <= (core_width / 2)
    power_upper = np.array([trapezoid(np.abs(field[upper_mask]) ** 2, x[upper_mask]) for field in e_evol])
    power_lower = np.array([trapezoid(np.abs(field[lower_mask]) ** 2, x[lower_mask]) for field in e_evol])

    power_upper_norm = power_upper / power_upper[0]
    power_lower_norm = power_lower / power_upper[0]
    lc_idx, lc = analyzer.estimate_coupling_length(z, power_lower_norm)

    return CouplerT3Result(
        x=x,
        z=z,
        n_profile=n_profile,
        n_effs=n_effs,
        modes=modes,
        e_evol=e_evol,
        power_upper=power_upper,
        power_lower=power_lower,
        power_upper_norm=power_upper_norm,
        power_lower_norm=power_lower_norm,
        lc_idx=lc_idx,
        lc=lc,
        gap=gap,
        core_width=core_width,
        guide_center=guide_center,
        lambda_launch=lambda_launch,
        length_total=length_total,
        sigma=sigma,
    )


def plot_coupler_t3_heatmap(result: CouplerT3Result, *, xlim_factor: float = 4.0):
    fig, ax = plt.subplots(figsize=(12, 8))
    intensity = np.abs(result.e_evol) ** 2
    im = ax.imshow(
        intensity,
        extent=[result.x[0] / um, result.x[-1] / um, result.z[-1] / um, 0],
        aspect="auto",
        cmap="magma",
        vmax=np.max(intensity) * 0.8,
    )
    fig.colorbar(im, label=r"Intensidade $|E|^2$")
    ax.set_title("BPM Test 3: Two-Waveguide Directional Coupler", fontweight="bold")
    ax.set_xlabel(r"Transverse Position $x$ [um]")
    ax.set_ylabel(r"Propagation Distance $z$ [um]")
    ax.set_xlim([-(xlim_factor) * result.guide_center / um, (xlim_factor) * result.guide_center / um])

    for edge in (
        -result.guide_center - result.core_width / 2,
        -result.guide_center + result.core_width / 2,
        result.guide_center - result.core_width / 2,
        result.guide_center + result.core_width / 2,
    ):
        ax.axvline(edge / um, color="white", ls="--", alpha=0.3)
    return fig, ax


def plot_parallel_coupler_heatmap(result: CouplerT3Result, *, xlim_factor: float = 4.0):
    return plot_coupler_t3_heatmap(result, xlim_factor=xlim_factor)


def plot_coupler_t3_profiles_and_transfer(result: CouplerT3Result, *, xlim_factor: float = 4.0):
    fig1, ax1 = plot_parallel_coupler_profiles(result, xlim_factor=xlim_factor)
    fig2, ax2 = plot_parallel_power_transfer(result)
    return (fig1, ax1), (fig2, ax2)


def plot_parallel_coupler_profiles(result: CouplerT3Result, *, xlim_factor: float = 4.0):
    input_profile = np.abs(result.e_evol[0, :])
    output_profile = np.abs(result.e_evol[-1, :])
    input_profile /= np.max(input_profile)
    output_profile /= np.max(output_profile)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(result.x / um, input_profile, "r--", lw=3, alpha=0.6, label="Input in the upper waveguide")
    ax1.plot(result.x / um, output_profile, "b-", lw=1.8, alpha=0.95, label="Output at z=L")
    ax1.axvspan(
        (-result.guide_center - result.core_width / 2) / um,
        (-result.guide_center + result.core_width / 2) / um,
        color="cyan",
        alpha=0.12,
    )
    ax1.axvspan(
        (result.guide_center - result.core_width / 2) / um,
        (result.guide_center + result.core_width / 2) / um,
        color="cyan",
        alpha=0.12,
    )
    ax1.set_title("Test 3: Coupler Input and Output Profiles", fontweight="bold")
    ax1.set_xlabel("Transverse Position x [um]")
    ax1.set_ylabel("Normalized Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-(xlim_factor) * result.guide_center / um, (xlim_factor) * result.guide_center / um])
    return fig1, ax1


def plot_parallel_power_transfer(result: CouplerT3Result):
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(result.z / um, result.power_upper_norm, lw=2, label="Upper waveguide")
    ax2.plot(result.z / um, result.power_lower_norm, lw=2, label="Lower waveguide")
    ax2.axvline(result.lc / um, color="k", ls="--", alpha=0.7, label=f"Lc = {result.lc / um:.2f} um")
    ax2.scatter([result.lc / um], [result.power_lower_norm[result.lc_idx]], color="k", s=35, zorder=3)
    ax2.set_title("Test 3: Power Transfer Between Waveguides", fontweight="bold")
    ax2.set_xlabel("Propagation Distance z [um]")
    ax2.set_ylabel("Normalized Power")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    return fig2, ax2


def plot_parallel_coupler_profiles_and_transfer(result: CouplerT3Result, *, xlim_factor: float = 4.0):
    return plot_coupler_t3_profiles_and_transfer(result, xlim_factor=xlim_factor)


def _run_bpm_propagation_silent(
    initial_field: np.ndarray,
    nz: int,
    n_points: int,
    alpha1: complex,
    alpha2: complex,
    t1: float,
    laplacian,
    identity,
    dz: float,
    n_profile: np.ndarray,
    n_effs: np.ndarray,
) -> np.ndarray:
    _ = laplacian, identity
    field_history = np.zeros((nz, n_points), dtype=complex)
    field_history[0, :] = initial_field

    lo_a, mid_a, up_a, lo_b, mid_b, up_b = crank_nicolson_tridiagonal_coefficients(
        alpha1, alpha2, t1, dz, n_profile, float(n_effs[0])
    )
    ab_a = pack_tridiagonal_banded(lo_a, mid_a, up_a)

    for i in range(1, nz):
        rhs = tridiagonal_matvec(lo_b, mid_b, up_b, field_history[i - 1, :])
        field_history[i, :] = solve_banded_tridiagonal(ab_a, rhs)
    return field_history


def sweep_gap_to_lc_parallel_coupler(
    analyzer: WaveguideModalBPM1D,
    *,
    x_base: np.ndarray,
    dx_base: float,
    refractive_indices: Sequence[float],
    thicknesses: Sequence[float],
    gap_values: Sequence[float] | None = None,
    lambda_launch: float = 1550.0 * nm,
    length_total: float = 2000.0 * um,
    sigma_ratio: float = 0.45,
    k_search: int = 6,
    extra_margin: float = 5.0 * um,
    min_fraction: float = 0.05,
    min_power: float = 0.05,
) -> list[dict[str, Any]]:
    if gap_values is None:
        gap_values = np.arange(0.1, 3.0 + 0.25, 0.25) * um
    gap_values = np.asarray(gap_values, dtype=float)

    core_width = _core_thickness_from_layers(thicknesses, refractive_indices)
    sigma = sigma_ratio * core_width
    x_half_base = np.max(np.abs(x_base))
    k0_local = 2 * np.pi / lambda_launch
    rows: list[dict[str, Any]] = []

    for gap in gap_values:
        guide_center = 0.5 * (core_width + gap)
        required_half = guide_center + 0.5 * core_width + extra_margin
        if required_half <= x_half_base:
            x_local = x_base.copy()
        else:
            n_points_local = int(np.ceil((2.0 * required_half) / dx_base)) + 1
            x_local = np.linspace(-required_half, required_half, n_points_local)

        dx_local = x_local[1] - x_local[0]
        n_points_local = len(x_local)

        n_profile_local = analyzer.build_dual_core_profile(
            x_vec=x_local,
            center_offset=guide_center,
            core_width=core_width,
            n_core=max(refractive_indices),
            n_clad=refractive_indices[0],
        )
        a_local, b_local = analyzer.assemble_helmholtz_operator(n_profile_local, dx_local, k0_local)
        n_effs_local, _ = analyzer.solve_guided_modes(
            a_local,
            b_local,
            n_core=max(refractive_indices),
            n_cladding=refractive_indices[0],
            k_search=min(k_search, n_points_local - 2),
        )

        if len(n_effs_local) == 0:
            rows.append(
                {
                    "gap": gap,
                    "gap_um": gap / um,
                    "Lc_bpm": np.nan,
                    "Lc_bpm_um": np.nan,
                    "Lc_supermode": np.nan,
                    "Lc_supermode_um": np.nan,
                    "peak_transfer": np.nan,
                    "peak_found": False,
                    "length_used": length_total,
                    "length_used_um": length_total / um,
                    "delta_neff": np.nan,
                    "n_points": n_points_local,
                    "window_um": (x_local[-1] - x_local[0]) / um,
                }
            )
            continue

        if len(n_effs_local) >= 2:
            delta_neff = abs(n_effs_local[0] - n_effs_local[1])
            lc_supermode = estimate_lc_by_supermodes(lambda_launch, n_effs_local[0], n_effs_local[1])
        else:
            delta_neff = np.nan
            lc_supermode = np.nan

        dz_local = lambda_launch / (4 * n_effs_local[0])
        nz_local = int(length_total / dz_local)
        alpha1_local = -1j / (2 * k0_local * n_effs_local[0])
        alpha2_local = -1j * k0_local / (2 * n_effs_local[0])
        identity_local = diags([np.ones(n_points_local)], [0], format="csr")
        laplacian_local = diags(
            [np.ones(n_points_local - 1), np.full(n_points_local, -2.0), np.ones(n_points_local - 1)],
            [-1, 0, 1],
            format="csr",
        )
        t1_local = 1.0 / dx_local**2

        e_launch_local = analyzer.generate_gaussian_input(x_local, xc=guide_center, sigma=sigma)
        initial_field_local = analyzer.normalize_field(x_local, e_launch_local)
        e_evol_local = _run_bpm_propagation_silent(
            initial_field_local,
            nz_local,
            n_points_local,
            alpha1_local,
            alpha2_local,
            t1_local,
            laplacian_local,
            identity_local,
            dz_local,
            n_profile_local,
            n_effs_local,
        )
        z_local = np.arange(nz_local) * dz_local
        lower_mask_local = np.abs(x_local + guide_center) <= (core_width / 2)
        upper_mask_local = np.abs(x_local - guide_center) <= (core_width / 2)
        power_upper = np.array(
            [trapezoid(np.abs(field[upper_mask_local]) ** 2, x_local[upper_mask_local]) for field in e_evol_local]
        )
        power_lower = np.array(
            [trapezoid(np.abs(field[lower_mask_local]) ** 2, x_local[lower_mask_local]) for field in e_evol_local]
        )
        power_lower_norm = power_lower / power_upper[0]
        peak_info = estimate_first_relevant_peak(
            z_local,
            power_lower_norm,
            min_fraction=min_fraction,
            min_power=min_power,
        )

        rows.append(
            {
                "gap": gap,
                "gap_um": gap / um,
                "Lc_bpm": peak_info["lc_from_start"],
                "Lc_bpm_um": peak_info["lc_from_start"] / um,
                "Lc_supermode": lc_supermode,
                "Lc_supermode_um": lc_supermode / um,
                "peak_transfer": peak_info["peak_value"],
                "peak_found": peak_info["peak_found"],
                "length_used": peak_info["length_used"],
                "length_used_um": peak_info["length_used"] / um,
                "delta_neff": delta_neff,
                "n_points": n_points_local,
                "window_um": (x_local[-1] - x_local[0]) / um,
            }
        )
    return rows


def plot_gap_to_lc_parallel_coupler(
    sweep_results: Sequence[dict[str, Any]],
    *,
    current_gap: float | None = None,
    current_lc: float | None = None,
):
    return plot_parameter_to_lc(
        sweep_results,
        parameter_key="gap",
        parameter_label="gap entre as guias",
        unit_scale=um,
        unit_label="um",
        title_prefix="Teste 3",
        current_parameter=current_gap,
        current_lc=current_lc,
    )


def plot_gap_to_lc_curve_parallel_coupler(
    sweep_results: Sequence[dict[str, Any]],
    *,
    current_gap: float | None = None,
    current_lc: float | None = None,
    ax=None,
):
    return plot_parameter_lc(
        sweep_results,
        parameter_key="gap",
        parameter_label="gap entre as guias",
        unit_scale=um,
        unit_label="um",
        title_prefix="Parallel coupler",
        current_parameter=current_gap,
        current_lc=current_lc,
        ax=ax,
    )


def plot_gap_to_peak_transfer_parallel_coupler(
    sweep_results: Sequence[dict[str, Any]],
    *,
    current_gap: float | None = None,
    ax=None,
):
    return plot_parameter_peak_transfer(
        sweep_results,
        parameter_key="gap",
        parameter_label="gap entre as guias",
        unit_scale=um,
        unit_label="um",
        title_prefix="Parallel coupler",
        current_parameter=current_gap,
        ax=ax,
    )


def summarize_coupler_output(result: CouplerSimulationResult) -> dict[str, float]:
    power_ref = result.power_total[0]
    upper_out = 100 * result.power_upper[-1] / power_ref
    lower_out = 100 * result.power_lower[-1] / power_ref
    loss_out = 100 * (result.power_total[-1] - result.power_upper[-1] - result.power_lower[-1]) / power_ref
    return {
        "upper_out_percent": float(upper_out),
        "lower_out_percent": float(lower_out),
        "outside_cores_percent": float(loss_out),
        "power_total_norm_final": float(result.power_total[-1] / power_ref),
        "dz_um": float(result.dz_local / um),
    }


def plot_coupler_power_transfer(
    result: CouplerSimulationResult,
    *,
    ax=None,
    unit_scale: float = um,
    unit_label: str = "um",
    title: str = "Power confined to the two output ports",
    upper_label: str = "Upper waveguide",
    lower_label: str = "Lower waveguide",
    total_label: str = "Total power",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    power_ref = result.power_total[0]
    ax.plot(result.z_vec / unit_scale, result.power_upper / power_ref, label=upper_label)
    ax.plot(result.z_vec / unit_scale, result.power_lower / power_ref, label=lower_label)
    ax.plot(result.z_vec / unit_scale, result.power_total / power_ref, "k--", lw=1.0, label=total_label)
    ax.set_title(title)
    ax.set_xlabel(f"z [{unit_label}]")
    ax.set_ylabel("Normalized power")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_input_output_profiles(
    *,
    x_vec: np.ndarray,
    field_input: np.ndarray,
    field_output: np.ndarray,
    spans: Sequence[dict[str, float | str]] | None = None,
    ax=None,
    unit_scale: float = um,
    unit_label: str = "um",
    title: str = "Profile comparison",
    input_label: str = "Input",
    output_label: str = "Output",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    profile_in = np.abs(field_input).astype(float)
    profile_out = np.abs(field_output).astype(float)
    if np.max(profile_in) > 0:
        profile_in /= np.max(profile_in)
    if np.max(profile_out) > 0:
        profile_out /= np.max(profile_out)

    ax.plot(x_vec / unit_scale, profile_in, "r--", lw=2.0, label=input_label)
    ax.plot(x_vec / unit_scale, profile_out, "b-", lw=1.6, label=output_label)

    if spans is not None:
        for span in spans:
            x_min = float(span["x_min"]) / unit_scale
            x_max = float(span["x_max"]) / unit_scale
            color = str(span.get("color", "orange"))
            alpha = float(span.get("alpha", 0.15))
            ax.axvspan(x_min, x_max, color=color, alpha=alpha)

    ax.set_title(title)
    ax.set_xlabel(f"Transverse position x [{unit_label}]")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def _coupler_terminal_spans(result: CouplerSimulationResult, params: CouplerTestParams) -> list[dict[str, float | str]]:
    offset_end = float(result.offset_trace[-1])
    half = float(params.core_width / 2)
    return [
        {
            "x_min": offset_end - half,
            "x_max": offset_end + half,
            "color": "orange",
            "alpha": 0.15,
        },
        {
            "x_min": -offset_end - half,
            "x_max": -offset_end + half,
            "color": "green",
            "alpha": 0.12,
        },
    ]


def plot_coupler_output_profiles(
    result: CouplerSimulationResult,
    params: CouplerTestParams,
    *,
    ax=None,
    unit_scale: float = um,
    unit_label: str = "um",
    title: str = "Profile comparison",
):
    return plot_input_output_profiles(
        x_vec=result.x_vec,
        field_input=result.e_evol[0, :],
        field_output=result.e_evol[-1, :],
        spans=_coupler_terminal_spans(result, params),
        ax=ax,
        unit_scale=unit_scale,
        unit_label=unit_label,
        title=title,
        input_label="Input",
        output_label="Output",
    )


def plot_coupler_heatmap(
    result: CouplerSimulationResult,
    params: CouplerTestParams,
    *,
    ax=None,
    unit_scale: float = um,
    unit_label: str = "um",
    title: str = "BPM: Gaussian beam in an optical coupler",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    intensity = np.abs(result.e_evol) ** 2
    im = ax.imshow(
        intensity,
        extent=[
            result.x_vec[0] / unit_scale,
            result.x_vec[-1] / unit_scale,
            result.z_vec[-1] / unit_scale,
            result.z_vec[0] / unit_scale,
        ],
        aspect="auto",
        cmap="magma",
        vmax=np.max(intensity) * 0.85,
    )
    fig.colorbar(im, ax=ax, label=r"Intensidade $|E|^2$")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(f"Transverse position x [{unit_label}]")
    ax.set_ylabel(f"Propagation distance z [{unit_label}]")

    for edge_curve in (
        result.offset_trace + params.core_width / 2,
        result.offset_trace - params.core_width / 2,
        -result.offset_trace + params.core_width / 2,
        -result.offset_trace - params.core_width / 2,
    ):
        ax.plot(edge_curve / unit_scale, result.z_vec / unit_scale, "w--", lw=0.8, alpha=0.8)

    ax.axhline(params.bend_length / unit_scale, color="cyan", ls=":", lw=1.0, alpha=0.9)
    ax.axhline(
        (params.bend_length + params.length_coupling) / unit_scale,
        color="cyan",
        ls=":",
        lw=1.0,
        alpha=0.9,
    )
    ax.set_xlim(result.x_vec[0] / unit_scale, result.x_vec[-1] / unit_scale)
    return fig, ax

def plot_coupler_transfer_and_profiles(result: CouplerSimulationResult, params: CouplerTestParams):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_coupler_power_transfer(result, ax=axes[0])
    plot_coupler_output_profiles(result, params, ax=axes[1])

    fig.tight_layout()
    return fig, axes

def sweep_coupler_wavelength_response(
    params: CouplerTestParams,
    lambda_values: Sequence[float],
    *,
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
) -> list[dict[str, float]]:
    lambda_values_arr = np.asarray(lambda_values, dtype=float)

    def _single(lambda_now: float) -> dict[str, float]:
        result = simulate_coupler(params=params, lambda_launch=float(lambda_now))
        summary = summarize_coupler_output(result)
        return {
            "lambda": float(lambda_now),
            "lambda_nm": float(lambda_now / nm),
            "upper": summary["upper_out_percent"],
            "lower": summary["lower_out_percent"],
            "loss": summary["outside_cores_percent"],
            "dz_um": summary["dz_um"],
        }

    if Parallel is None or delayed is None:
        return [_single(lambda_now) for lambda_now in lambda_values_arr]

    return Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        batch_size="auto",
    )(delayed(_single)(lambda_now) for lambda_now in lambda_values_arr)


def plot_coupler_wavelength_response(
    lambda_values: Sequence[float],
    sweep_results: Sequence[dict[str, float]],
    *,
    lambda_ref: float | None = None,
    ax=None,
    unit_scale: float = nm,
    unit_label: str = "nm",
    title: str = "Spectral response of the output ports",
):
    lambda_arr = np.asarray(lambda_values, dtype=float)
    if lambda_arr.size == 0:
        raise ValueError("lambda_values cannot be empty.")

    upper = np.array([item["upper"] for item in sweep_results], dtype=float)
    lower = np.array([item["lower"] for item in sweep_results], dtype=float)
    loss = np.array([item["loss"] for item in sweep_results], dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))
    else:
        fig = ax.figure

    ax.plot(lambda_arr / unit_scale, upper, "o-", label="Upper port")
    ax.plot(lambda_arr / unit_scale, lower, "s-", label="Lower port")
    ax.plot(lambda_arr / unit_scale, loss, "k--", lw=1.2, label="Outside the cores")
    if lambda_ref is not None:
        ax.axvline(lambda_ref / unit_scale, color="gray", ls=":", lw=1.0, label="current lambda")

    # Mantem o eixo X no intervalo realmente varrido, sem ser expandido pela linha de referencia.
    x_min = float(np.min(lambda_arr) / unit_scale)
    x_max = float(np.max(lambda_arr) / unit_scale)
    ax.set_xlim(x_min, x_max)

    ax.set_title(title)
    ax.set_xlabel(f"Input wavelength [{unit_label}]")
    ax.set_ylabel("Output power [%]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax
