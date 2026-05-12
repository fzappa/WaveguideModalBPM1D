"""Generate `figures/16_sweep_parameter_lc.png` - `plot_parameter_lc`.

Parameters: `example_parameters.py`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from example_parameters import (
    PLOT_PARAMETER_KEY,
    PLOT_PARAMETER_LABEL,
    PLOT_PARAMETER_UNIT_LABEL,
    PLOT_PARAMETER_UNIT_SCALE,
    PLOT_SWEEP_PARAMETER_LC_TITLE_PREFIX,
    SWEEP_LC_CURRENT_LC,
    SWEEP_LC_CURRENT_PARAMETER,
    SWEEP_LC_PLOT_AX,
    SWEEP_COUPLER_LAMBDA_LAUNCH,
    SWEEP_COUPLER_MIN_FRACTION,
    SWEEP_COUPLER_MIN_POWER,
    coupler_example_params,
    gap_values_coupler_sweep,
)
from figure_helpers import ensure_repo_on_path, save_figure, silent_stdout

ensure_repo_on_path()

from WaveguideModalBPM1D import plot_parameter_lc, sweep_gap_to_lc_coupler


def main() -> None:
    params = coupler_example_params()
    with silent_stdout():
        sweep = sweep_gap_to_lc_coupler(
            params=params,
            gap_values=gap_values_coupler_sweep(),
            lambda_launch=SWEEP_COUPLER_LAMBDA_LAUNCH,
            min_fraction=SWEEP_COUPLER_MIN_FRACTION,
            min_power=SWEEP_COUPLER_MIN_POWER,
        )
    fig, _ = plot_parameter_lc(
        sweep,
        parameter_key=PLOT_PARAMETER_KEY,
        parameter_label=PLOT_PARAMETER_LABEL,
        unit_scale=PLOT_PARAMETER_UNIT_SCALE,
        unit_label=PLOT_PARAMETER_UNIT_LABEL,
        title_prefix=PLOT_SWEEP_PARAMETER_LC_TITLE_PREFIX,
        current_parameter=SWEEP_LC_CURRENT_PARAMETER,
        current_lc=SWEEP_LC_CURRENT_LC,
        ax=SWEEP_LC_PLOT_AX,
    )
    save_figure(fig, "16_sweep_parameter_lc.png")


if __name__ == "__main__":
    main()
