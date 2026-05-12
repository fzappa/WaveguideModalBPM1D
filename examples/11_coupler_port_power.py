"""Generate `figures/11_coupler_port_power.png` - `plot_coupler_power_transfer`.

Parameters: `example_parameters.py`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from example_parameters import (
    COUPLER_POWER_AX,
    COUPLER_POWER_LOWER_LABEL,
    COUPLER_POWER_TITLE,
    COUPLER_POWER_TOTAL_LABEL,
    COUPLER_POWER_UNIT_LABEL,
    COUPLER_POWER_UNIT_SCALE,
    COUPLER_POWER_UPPER_LABEL,
    SIMULATE_COUPLER_LAMBDA_LAUNCH,
    coupler_example_params,
)
from figure_helpers import ensure_repo_on_path, save_figure, silent_stdout

ensure_repo_on_path()

from WaveguideModalBPM1D import plot_coupler_power_transfer, simulate_coupler


def main() -> None:
    params = coupler_example_params()
    with silent_stdout():
        sim = simulate_coupler(params=params, lambda_launch=SIMULATE_COUPLER_LAMBDA_LAUNCH)
    fig, _ = plot_coupler_power_transfer(
        sim,
        ax=COUPLER_POWER_AX,
        unit_scale=COUPLER_POWER_UNIT_SCALE,
        unit_label=COUPLER_POWER_UNIT_LABEL,
        title=COUPLER_POWER_TITLE,
        upper_label=COUPLER_POWER_UPPER_LABEL,
        lower_label=COUPLER_POWER_LOWER_LABEL,
        total_label=COUPLER_POWER_TOTAL_LABEL,
    )
    save_figure(fig, "11_coupler_port_power.png")


if __name__ == "__main__":
    main()
