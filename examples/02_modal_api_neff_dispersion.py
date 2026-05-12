"""Generate `figures/02_modal_api_neff_dispersion.png` - `plot_dispersion_curves` (modal API).

Geometry and d/lambda grid: `example_parameters.py`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from example_parameters import (
    MODAL_DISPERSION_MAX_MODES_CAP,
    modal_dispersion_d_norm_vec,
    modal_structure_02,
)
from figure_helpers import ensure_repo_on_path, save_figure

ensure_repo_on_path()

from WaveguideModalBPM1D import compute_dispersion_curves, plot_dispersion_curves


def main() -> None:
    res = compute_dispersion_curves(
        modal_structure_02(),
        d_norm_vec=modal_dispersion_d_norm_vec(),
        max_modes_cap=MODAL_DISPERSION_MAX_MODES_CAP,
    )
    fig, _ = plot_dispersion_curves(res)
    save_figure(fig, "02_modal_api_neff_dispersion.png")


if __name__ == "__main__":
    main()
