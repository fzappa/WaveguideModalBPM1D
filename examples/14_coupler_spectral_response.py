"""Generate `figures/14_coupler_spectral_response.png` - `plot_coupler_wavelength_response`.

Parameters: `example_parameters.py`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from example_parameters import (
    WAVELENGTH_PLOT_AX,
    WAVELENGTH_PLOT_LAMBDA_REF,
    WAVELENGTH_PLOT_TITLE,
    WAVELENGTH_PLOT_UNIT_LABEL,
    WAVELENGTH_PLOT_UNIT_SCALE,
    WAVELENGTH_SWEEP_BACKEND,
    WAVELENGTH_SWEEP_N_JOBS,
    WAVELENGTH_SWEEP_VERBOSE,
    coupler_example_params,
    wavelength_values_sweep,
)
from figure_helpers import ensure_repo_on_path, save_figure, silent_stdout

ensure_repo_on_path()

from WaveguideModalBPM1D import plot_coupler_wavelength_response, sweep_coupler_wavelength_response


def main() -> None:
    params = coupler_example_params()
    lambdas = wavelength_values_sweep()
    with silent_stdout():
        rows = sweep_coupler_wavelength_response(
            params,
            lambdas,
            n_jobs=WAVELENGTH_SWEEP_N_JOBS,
            backend=WAVELENGTH_SWEEP_BACKEND,
            verbose=WAVELENGTH_SWEEP_VERBOSE,
        )
    fig, _ = plot_coupler_wavelength_response(
        lambdas,
        rows,
        lambda_ref=WAVELENGTH_PLOT_LAMBDA_REF,
        ax=WAVELENGTH_PLOT_AX,
        unit_scale=WAVELENGTH_PLOT_UNIT_SCALE,
        unit_label=WAVELENGTH_PLOT_UNIT_LABEL,
        title=WAVELENGTH_PLOT_TITLE,
    )
    save_figure(fig, "14_coupler_spectral_response.png")


if __name__ == "__main__":
    main()
