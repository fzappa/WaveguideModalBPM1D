from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from .constants import nm, um
from .parameters import ModalStructureParams


@dataclass(frozen=True)
class ModalAnalysisResult:
    x: np.ndarray
    dx: float
    boundaries: np.ndarray
    n_profile: np.ndarray
    n_effs: np.ndarray
    modes: np.ndarray
    params: ModalStructureParams


@dataclass(frozen=True)
class DispersionAnalysisResult:
    d_norm_vec: np.ndarray
    matriz_neff: np.ndarray
    max_modes: int
    params: ModalStructureParams


def build_index_profile(
    x_vec: np.ndarray, indices: tuple[float, ...], boundaries: np.ndarray
) -> np.ndarray:
    conds = [x_vec <= boundaries[0]]
    for i in range(len(boundaries) - 1):
        conds.append((x_vec > boundaries[i]) & (x_vec <= boundaries[i + 1]))
    return np.piecewise(x_vec, conds, indices)


def assemble_helmholtz_operator(n_profile: np.ndarray, dx: float, k0: float):
    n_points = len(n_profile)
    laplacian = diags(
        [np.ones(n_points - 1), np.full(n_points, -2.0), np.ones(n_points - 1)],
        [-1, 0, 1],
        format="csr",
    )
    a_mat = (1.0 / dx**2) * laplacian + diags([k0**2 * n_profile**2], [0], format="csr")
    b_mat = diags([np.full(n_points, k0**2)], [0], format="csr")
    return a_mat, b_mat


def solve_guided_modes(
    a_mat,
    b_mat,
    n_core: float,
    n_cladding: float,
    k_search: int = 10,
):
    eigenvalues, eigenvectors = eigsh(
        a_mat,
        k=k_search,
        M=b_mat,
        sigma=n_core**2,
        which="LM",
    )
    mask = (eigenvalues > n_cladding**2) & (eigenvalues < n_core**2 + 0.01)
    if not np.any(mask):
        return np.array([]), np.zeros((a_mat.shape[0], 0))

    neff2 = eigenvalues[mask].real
    order = np.argsort(neff2)[::-1]
    return np.sqrt(neff2[order]), eigenvectors[:, mask][:, order]


def analyze_modes(params: ModalStructureParams) -> ModalAnalysisResult:
    width = sum(params.thicknesses)
    boundaries = np.cumsum(params.thicknesses) - width / 2
    x = np.linspace(-width / 2, width / 2, params.n_points)
    dx = width / (params.n_points - 1)
    k0 = 2 * np.pi / params.lambda_0
    n_profile = build_index_profile(x, params.refractive_indices, boundaries)
    a_mat, b_mat = assemble_helmholtz_operator(n_profile, dx, k0)
    n_effs, modes = solve_guided_modes(
        a_mat,
        b_mat,
        n_core=params.n_core,
        n_cladding=params.n_clad,
    )
    return ModalAnalysisResult(
        x=x,
        dx=dx,
        boundaries=boundaries,
        n_profile=n_profile,
        n_effs=n_effs,
        modes=modes,
        params=params,
    )


def compute_dispersion_curves(
    params: ModalStructureParams,
    d_norm_vec: np.ndarray | None = None,
    max_modes_cap: int = 10,
) -> DispersionAnalysisResult:
    if d_norm_vec is None:
        d_norm_vec = np.linspace(0.1, 5.0, 60)

    k0 = 2 * np.pi / params.lambda_0
    na = np.sqrt(params.n_core**2 - params.n_clad**2)
    v_max = k0 * (d_norm_vec[-1] * params.lambda_0 / 2) * na
    max_modes = min(int(np.ceil(2 * v_max / np.pi)) + 1, max_modes_cap)
    neff_matrix = np.full((max_modes, len(d_norm_vec)), np.nan)
    core_idx = int(np.argmax(params.refractive_indices))

    for i, d_norm in enumerate(d_norm_vec):
        local_thicknesses = list(params.thicknesses)
        local_thicknesses[core_idx] = d_norm * params.lambda_0
        width = sum(local_thicknesses)
        boundaries = np.cumsum(local_thicknesses) - width / 2
        x = np.linspace(-width / 2, width / 2, params.n_points)
        dx = width / (params.n_points - 1)
        n_profile = build_index_profile(x, params.refractive_indices, boundaries)

        try:
            a_mat, b_mat = assemble_helmholtz_operator(n_profile, dx, k0)
            local_neffs, _ = solve_guided_modes(
                a_mat,
                b_mat,
                params.n_core,
                params.n_clad,
                k_search=min(max_modes + 2, params.n_points - 2),
            )
        except Exception:
            continue

        for mode_idx in range(min(len(local_neffs), max_modes)):
            neff_matrix[mode_idx, i] = local_neffs[mode_idx]

    return DispersionAnalysisResult(
        d_norm_vec=np.asarray(d_norm_vec),
        matriz_neff=neff_matrix,
        max_modes=max_modes,
        params=params,
    )


def plot_modal_analysis(result: ModalAnalysisResult):
    n_modes = len(result.n_effs)
    if n_modes == 0:
        raise ValueError("No guided mode was found to plot.")

    fig, axes = plt.subplots(
        nrows=n_modes + 1,
        ncols=1,
        figsize=(8, 2 * (n_modes + 2)),
        sharex=True,
    )

    axes[0].plot(result.x / um, result.n_profile, "k-", lw=1, alpha=0.8)
    axes[0].set_ylabel("Index n")
    axes[0].set_title(
        f"Multilayer Profile and TE/TM Modes (lambda = {result.params.lambda_0 / nm:.0f} nm)"
    )

    cmap = plt.colormaps.get_cmap("Pastel1")
    color_idx = np.linspace(0, 1, len(result.params.refractive_indices))
    x_start = -sum(result.params.thicknesses) / 2
    for i, (_, thickness) in enumerate(
        zip(result.params.refractive_indices, result.params.thicknesses)
    ):
        x_end = x_start + thickness
        axes[0].fill_between(
            [x_start / um, x_end / um],
            [min(result.params.refractive_indices) - 0.01] * 2,
            [max(result.params.refractive_indices) + 0.01] * 2,
            color=cmap(color_idx[i]),
            alpha=0.3,
        )
        x_start = x_end

    for b in result.boundaries[:-1]:
        axes[0].axvline(b / um, color="gray", lw=0.5, ls=":")

    for i in range(n_modes):
        ax = axes[i + 1]
        norm_field = result.modes[:, i] / np.max(np.abs(result.modes[:, i]))
        ax.plot(result.x / um, norm_field, label=f"TE_{i} / TM_{i} | neff={result.n_effs[i]:.4f}")
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        for b in result.boundaries[:-1]:
            ax.axvline(b / um, color="red", ls=":", alpha=0.3)
        ax.set_ylabel(f"Mode {i}")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("x [um]")
    fig.tight_layout()
    return fig, axes


def plot_dispersion_curves(result: DispersionAnalysisResult):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title(
        f"TE/TM Modal Dispersion Curves (lambda = {result.params.lambda_0 / nm:.0f} nm)"
    )

    for mode_idx in range(result.max_modes):
        mask = ~np.isnan(result.matriz_neff[mode_idx, :])
        d_plot = result.d_norm_vec[mask]
        n_plot = result.matriz_neff[mode_idx, mask]
        if len(d_plot) == 0:
            continue
        line, = ax1.plot(d_plot, n_plot, lw=1)
        if len(d_plot) > 10:
            idx_text = len(d_plot) // 3
            ax1.text(
                d_plot[idx_text] - 0.2,
                n_plot[idx_text] + 0.002,
                f"TE_{mode_idx}, TM_{mode_idx}",
                fontsize=10,
                rotation=35,
                ha="center",
                color=line.get_color(),
            )

    ax1.set_xlabel("d / lambda")
    ax1.set_xlim(0, max(result.d_norm_vec))
    ax1.set_ylabel("Effective index neff")
    ax1.set_ylim(result.params.n_clad, result.params.n_core)
    return fig, ax1


AnaliseModalResultado = ModalAnalysisResult
AnaliseDispersaoResultado = DispersionAnalysisResult
perfil_indice = build_index_profile
montar_operador_helmholtz = assemble_helmholtz_operator
resolver_modos_guiados = solve_guided_modes
analisar_modos = analyze_modes
calcular_curvas_dispersao = compute_dispersion_curves
plotar_analise_modal = plot_modal_analysis
plotar_curvas_dispersao = plot_dispersion_curves
