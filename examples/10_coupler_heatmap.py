"""Generate `figures/10_coupler_heatmap.png` - `plot_coupler_heatmap`.

Parameters: `example_parameters.py`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from example_parameters import (
    COUPLER_HEATMAP_AX,
    COUPLER_HEATMAP_TITLE,
    COUPLER_HEATMAP_UNIT_LABEL,
    COUPLER_HEATMAP_UNIT_SCALE,
    SIMULATE_COUPLER_LAMBDA_LAUNCH,
    coupler_example_params,
)
from figure_helpers import ensure_repo_on_path, save_figure, silent_stdout

ensure_repo_on_path()

from WaveguideModalBPM1D import plot_coupler_heatmap, simulate_coupler


def main() -> None:
    params = coupler_example_params()
    with silent_stdout():
        sim = simulate_coupler(params=params, lambda_launch=SIMULATE_COUPLER_LAMBDA_LAUNCH)
    fig, _ = plot_coupler_heatmap(
        sim,
        params,
        ax=COUPLER_HEATMAP_AX,
        unit_scale=COUPLER_HEATMAP_UNIT_SCALE,
        unit_label=COUPLER_HEATMAP_UNIT_LABEL,
        title=COUPLER_HEATMAP_TITLE,
    )
    save_figure(fig, "10_coupler_heatmap.png")


if __name__ == "__main__":
    main()
