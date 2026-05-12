from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence

import numpy as np

from .constants import um
from .parameters import CouplerTestParams
from .test_cases import simulate_coupler


def estimate_first_relevant_peak(
    z_vec: Sequence[float],
    power_trace: Sequence[float],
    *,
    z_start: float | None = None,
    z_end: float | None = None,
    min_fraction: float = 0.05,
    min_power: float = 0.05,
) -> dict[str, Any]:
    z_arr = np.asarray(z_vec, dtype=float)
    p_arr = np.asarray(power_trace, dtype=float)

    if len(z_arr) != len(p_arr):
        raise ValueError("z_vec and power_trace must have the same length.")
    if len(z_arr) < 2:
        raise ValueError("z_vec must contain at least 2 points.")

    z_start = float(z_arr[0] if z_start is None else z_start)
    z_end = float(z_arr[-1] if z_end is None else z_end)
    if z_end < z_start:
        raise ValueError("z_end must be greater than or equal to z_start.")

    mask = (z_arr >= z_start) & (z_arr <= z_end)
    if np.count_nonzero(mask) < 5:
        return {
            "peak_found": False,
            "peak_index_region": None,
            "peak_index_global": None,
            "peak_value": np.nan,
            "z_peak_absolute": np.nan,
            "lc_from_start": np.nan,
            "length_used": np.nan,
            "z_window_start": z_start,
            "z_window_end": z_end,
        }

    z_region = z_arr[mask]
    p_region = p_arr[mask]
    idx_global = np.where(mask)[0]

    delta_power = np.diff(p_region)
    peak_candidates = np.where((delta_power[:-1] > 0) & (delta_power[1:] <= 0))[0] + 1

    min_start_idx = max(3, int(min_fraction * len(z_region)))
    min_start_idx = min(min_start_idx, len(z_region) - 1)

    valid_peaks = [
        idx for idx in peak_candidates if idx >= min_start_idx and p_region[idx] > min_power
    ]
    if valid_peaks:
        peak_idx = int(valid_peaks[0])
        peak_found = True
    else:
        peak_idx = int(min_start_idx + np.argmax(p_region[min_start_idx:]))
        peak_found = False

    z_peak_abs = float(z_region[peak_idx])
    return {
        "peak_found": peak_found,
        "peak_index_region": peak_idx,
        "peak_index_global": int(idx_global[peak_idx]),
        "peak_value": float(p_region[peak_idx]),
        "z_peak_absolute": z_peak_abs,
        "lc_from_start": float(z_peak_abs - z_start),
        "length_used": float(z_region[-1] - z_start),
        "z_window_start": z_start,
        "z_window_end": z_end,
    }


def estimate_lc_by_supermodes(lambda_launch: float, n_eff_1: float, n_eff_2: float) -> float:
    delta_neff = abs(float(n_eff_1) - float(n_eff_2))
    if delta_neff <= 0:
        return np.nan
    k0_local = 2.0 * np.pi / float(lambda_launch)
    return float(np.pi / (k0_local * delta_neff))


def sweep_parameter_to_lc(
    parameter_values: Sequence[float],
    simulate_fn: Callable[[float], Any],
    *,
    get_z: Callable[[Any], Sequence[float]],
    get_power_coupled: Callable[[Any], Sequence[float]],
    get_power_reference: Callable[[Any], float | Sequence[float]],
    parameter_label: str = "parameter",
    get_lambda: Callable[[Any], float] | None = None,
    get_neffs: Callable[[Any], Sequence[float]] | None = None,
    get_window: Callable[[Any], tuple[float, float] | None] | None = None,
    postprocess: Callable[[float, Any], dict[str, Any]] | None = None,
    min_fraction: float = 0.05,
    min_power: float = 0.05,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parameter_value in parameter_values:
        result = simulate_fn(float(parameter_value))

        z_arr = np.asarray(get_z(result), dtype=float)
        power_coupled_arr = np.asarray(get_power_coupled(result), dtype=float)

        power_ref_raw = get_power_reference(result)
        if np.isscalar(power_ref_raw):
            power_ref = float(power_ref_raw)
        else:
            power_ref = float(np.asarray(power_ref_raw, dtype=float)[0])
        if power_ref == 0:
            raise ValueError("The reference power cannot be zero.")

        power_norm = power_coupled_arr / power_ref

        z_start, z_end = z_arr[0], z_arr[-1]
        if get_window is not None:
            window = get_window(result)
            if window is not None:
                z_start, z_end = float(window[0]), float(window[1])

        peak_info = estimate_first_relevant_peak(
            z_arr,
            power_norm,
            z_start=z_start,
            z_end=z_end,
            min_fraction=min_fraction,
            min_power=min_power,
        )

        delta_neff = np.nan
        lc_supermode = np.nan
        if get_lambda is not None and get_neffs is not None:
            n_effs = np.asarray(get_neffs(result), dtype=float)
            if len(n_effs) >= 2:
                delta_neff = float(abs(n_effs[0] - n_effs[1]))
                lc_supermode = estimate_lc_by_supermodes(
                    get_lambda(result),
                    n_effs[0],
                    n_effs[1],
                )

        row = {
            parameter_label: float(parameter_value),
            "Lc_bpm": peak_info["lc_from_start"],
            "Lc_supermode": lc_supermode,
            "peak_transfer": peak_info["peak_value"],
            "peak_found": peak_info["peak_found"],
            "length_used": peak_info["length_used"],
            "delta_neff": delta_neff,
            "z_peak_absolute": peak_info["z_peak_absolute"],
            "z_window_start": peak_info["z_window_start"],
            "z_window_end": peak_info["z_window_end"],
        }
        if postprocess is not None:
            row.update(postprocess(float(parameter_value), result))

        rows.append(row)

    return rows


def plot_parameter_to_lc(
    sweep_results: Sequence[dict[str, Any]],
    *,
    parameter_key: str = "parameter",
    parameter_label: str = "Parameter",
    unit_scale: float = 1.0,
    unit_label: str = "",
    title_prefix: str = "Varredura",
    current_parameter: float | None = None,
    current_lc: float | None = None,
    figsize: tuple[float, float] = (14, 5),
):
    import matplotlib.pyplot as plt

    if len(sweep_results) == 0:
        raise ValueError("sweep_results cannot be empty.")

    (
        x_vals,
        lc_bpm,
        lc_super,
        peak_transfer,
        peak_found,
        length_used,
    ) = _extract_plot_arrays(
        sweep_results,
        parameter_key=parameter_key,
        unit_scale=unit_scale,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_parameter_lc(
        sweep_results,
        parameter_key=parameter_key,
        parameter_label=parameter_label,
        unit_scale=unit_scale,
        unit_label=unit_label,
        title_prefix=title_prefix,
        current_parameter=current_parameter,
        current_lc=current_lc,
        ax=axes[0],
    )

    plot_parameter_peak_transfer(
        sweep_results,
        parameter_key=parameter_key,
        parameter_label=parameter_label,
        unit_scale=unit_scale,
        unit_label=unit_label,
        title_prefix=title_prefix,
        current_parameter=current_parameter,
        ax=axes[1],
    )

    plt.tight_layout()
    return fig, axes


def _extract_plot_arrays(
    sweep_results: Sequence[dict[str, Any]],
    *,
    parameter_key: str,
    unit_scale: float,
):
    x_vals = np.array([row[parameter_key] for row in sweep_results], dtype=float) / unit_scale
    lc_bpm = np.array([row["Lc_bpm"] for row in sweep_results], dtype=float) / unit_scale
    lc_super = np.array([row["Lc_supermode"] for row in sweep_results], dtype=float) / unit_scale
    peak_transfer = np.array([row["peak_transfer"] for row in sweep_results], dtype=float)
    peak_found = np.array([row["peak_found"] for row in sweep_results], dtype=bool)
    length_used = np.array([row["length_used"] for row in sweep_results], dtype=float) / unit_scale
    return x_vals, lc_bpm, lc_super, peak_transfer, peak_found, length_used


def plot_parameter_lc(
    sweep_results: Sequence[dict[str, Any]],
    *,
    parameter_key: str = "parameter",
    parameter_label: str = "Parameter",
    unit_scale: float = 1.0,
    unit_label: str = "",
    title_prefix: str = "Varredura",
    current_parameter: float | None = None,
    current_lc: float | None = None,
    ax=None,
):
    import matplotlib.pyplot as plt

    if len(sweep_results) == 0:
        raise ValueError("sweep_results nao pode ser vazio.")

    x_vals, lc_bpm, lc_super, _, peak_found, length_used = _extract_plot_arrays(
        sweep_results,
        parameter_key=parameter_key,
        unit_scale=unit_scale,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ax.plot(
        x_vals,
        lc_super,
        marker="o",
        lw=2,
        label=r"$L_c$ from supermodes",
    )
    ax.plot(
        x_vals[peak_found],
        lc_bpm[peak_found],
        marker="s",
        lw=1.8,
        label=r"$L_c$ from BPM (first peak)",
    )

    if np.any(~peak_found):
        ax.scatter(
            x_vals[~peak_found],
            length_used[~peak_found],
            marker="^",
            s=70,
            facecolors="none",
            edgecolors="crimson",
            label="BPM without a peak in the simulated length",
        )

    if current_parameter is not None:
        ax.axvline(
            current_parameter / unit_scale,
            color="k",
            ls="--",
            alpha=0.7,
            label=f"current = {current_parameter / unit_scale:.2f} {unit_label}",
        )
    if current_lc is not None:
        ax.axhline(
            current_lc / unit_scale,
            color="gray",
            ls=":",
            alpha=0.8,
            label=f"current Lc = {current_lc / unit_scale:.2f} {unit_label}",
        )

    ax.set_title(f"{title_prefix}: parameter -> Lc", fontweight="bold")
    ax.set_xlabel(f"{parameter_label} [{unit_label}]")
    ax.set_ylabel(f"Coupling length Lc [{unit_label}]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_parameter_peak_transfer(
    sweep_results: Sequence[dict[str, Any]],
    *,
    parameter_key: str = "parameter",
    parameter_label: str = "Parameter",
    unit_scale: float = 1.0,
    unit_label: str = "",
    title_prefix: str = "Varredura",
    current_parameter: float | None = None,
    ax=None,
):
    import matplotlib.pyplot as plt

    if len(sweep_results) == 0:
        raise ValueError("sweep_results cannot be empty.")

    x_vals, _, _, peak_transfer, _, _ = _extract_plot_arrays(
        sweep_results,
        parameter_key=parameter_key,
        unit_scale=unit_scale,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ax.plot(
        x_vals,
        peak_transfer,
        marker="o",
        lw=2,
        color="tab:green",
        label="Transfer peak (BPM)",
    )
    if current_parameter is not None:
        ax.axvline(
            current_parameter / unit_scale,
            color="k",
            ls="--",
            alpha=0.7,
            label=f"current = {current_parameter / unit_scale:.2f} {unit_label}",
        )
    ax.set_title(f"{title_prefix}: transferred power peak", fontweight="bold")
    ax.set_xlabel(f"{parameter_label} [{unit_label}]")
    ax.set_ylabel("Normalized power in the opposite waveguide")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def sweep_gap_to_lc_coupler(
    params: CouplerTestParams | None = None,
    gap_values: Sequence[float] | None = None,
    *,
    lambda_launch: float | None = None,
    min_fraction: float = 0.05,
    min_power: float = 0.05,
) -> list[dict[str, Any]]:
    params = CouplerTestParams() if params is None else params
    if gap_values is None:
        gap_values = np.arange(0.10, 3.00 + 0.25, 0.25) * um
    gap_values_arr = np.asarray(gap_values, dtype=float)

    def _simulate_one(gap_value: float) -> dict[str, Any]:
        center_offset = 0.5 * (params.core_width + gap_value)
        params_local = replace(params, center_offset=center_offset)
        lambda_now = params.lambda_launch if lambda_launch is None else lambda_launch
        result = simulate_coupler(params=params_local, lambda_launch=lambda_now)
        return {"result": result, "params_local": params_local}

    rows = sweep_parameter_to_lc(
        gap_values_arr,
        _simulate_one,
        parameter_label="gap",
        get_z=lambda payload: payload["result"].z_vec,
        get_power_coupled=lambda payload: payload["result"].power_lower,
        get_power_reference=lambda payload: payload["result"].power_total[0],
        get_lambda=lambda payload: payload["result"].lambda_launch,
        get_neffs=lambda payload: payload["result"].n_effs_local,
        get_window=lambda payload: (
            payload["params_local"].bend_length,
            payload["params_local"].bend_length + payload["params_local"].length_coupling,
        ),
        postprocess=lambda gap_value, payload: {
            "gap_um": gap_value / um,
            "center_offset": payload["params_local"].center_offset,
            "center_offset_um": payload["params_local"].center_offset / um,
        },
        min_fraction=min_fraction,
        min_power=min_power,
    )

    for row in rows:
        row["Lc_bpm_um"] = row["Lc_bpm"] / um
        row["Lc_supermode_um"] = row["Lc_supermode"] / um
        row["length_used_um"] = row["length_used"] / um

    return rows


def plot_gap_to_lc_coupler(
    sweep_results: Sequence[dict[str, Any]],
    *,
    current_gap: float | None = None,
    current_lc: float | None = None,
    title_prefix: str = "Coupler",
    figsize: tuple[float, float] = (14, 5),
):
    return plot_parameter_to_lc(
        sweep_results,
        parameter_key="gap",
        parameter_label="gap in the coupling section",
        unit_scale=um,
        unit_label="um",
        title_prefix=title_prefix,
        current_parameter=current_gap,
        current_lc=current_lc,
        figsize=figsize,
    )

def plot_gap_to_lc_curve_coupler(
    sweep_results: Sequence[dict[str, Any]],
    *,
    current_gap: float | None = None,
    current_lc: float | None = None,
    title_prefix: str = "Coupler",
    ax=None,
):
    return plot_parameter_lc(
        sweep_results,
        parameter_key="gap",
        parameter_label="gap in the coupling section",
        unit_scale=um,
        unit_label="um",
        title_prefix=title_prefix,
        current_parameter=current_gap,
        current_lc=current_lc,
        ax=ax,
    )


def plot_gap_to_peak_transfer_coupler(
    sweep_results: Sequence[dict[str, Any]],
    *,
    current_gap: float | None = None,
    title_prefix: str = "Coupler",
    ax=None,
):
    return plot_parameter_peak_transfer(
        sweep_results,
        parameter_key="gap",
        parameter_label="gap in the coupling section",
        unit_scale=um,
        unit_label="um",
        title_prefix=title_prefix,
        current_parameter=current_gap,
        ax=ax,
    )
