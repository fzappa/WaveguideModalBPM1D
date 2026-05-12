# WaveguideModalBPM1D

> **TL;DR**: Python scientific engine for 1D modal analysis and BPM simulations of optical waveguides and couplers.

Python package for 1D modal analysis and Beam Propagation Method (BPM) simulations.

## What the package includes

- finite-difference modal solver for multilayer structures;
- BPM routines for straight waveguides and couplers;
- helper functions for parametric sweeps;
- visualization functions based on `matplotlib`.

## Repository structure

- `WaveguideModalBPM1D/`: package source code;
- `tests/`: basic automated tests;
- `examples/`: one script per figure (`NN_*.py`), `example_parameters.py` (all gallery numerics explicit), `figure_helpers.py`, `generate_figures.py`;
- `pyproject.toml`: packaging configuration.

## Example figures (`examples/`)

There are **22 numbered scripts** (`01_*.py` … `22_*.py`). Each configures the `Agg` backend, imports explicit parameters from `examples/example_parameters.py`, runs the minimal simulation for that plot, and writes one PNG under `examples/figures/`. That module lists every wavelength, index stack, grid size, sweep limit, and plot label used in the gallery so the scripts do **not** rely on omitting arguments to the library dataclasses (see **Package defaults** below for what the library would use if you omitted fields). Shared helpers live in `examples/figure_helpers.py` (`save_figure`, optional `silent_stdout`; path setup is also done inside `example_parameters.py` so imports resolve when you run a script from the repo root).

Run **one** example from the repository root:

```bash
pip install -e .
python examples/01_modal_api_profile_and_modes.py
```

Regenerate **all** PNGs in sequence:

```bash
python examples/generate_figures.py
```

| Script | Output PNG | Primary API (same layout where noted) |
|--------|------------|----------------------------------------|
| `01_modal_api_profile_and_modes.py` | `01_modal_api_profile_and_modes.png` | `plot_modal_analysis` |
| `02_modal_api_neff_dispersion.py` | `02_modal_api_neff_dispersion.png` | `plot_dispersion_curves` |
| `03_workflow_multilayer_modes.py` | `03_workflow_multilayer_modes.png` | `plot_multilayer_modes` |
| `04_workflow_dispersion_with_angle_axis.py` | `04_workflow_dispersion_with_angle_axis.png` | `plot_dispersion_with_angle_axis` |
| `05_straight_bpm_fundamental_heatmap.py` | `05_straight_bpm_fundamental_heatmap.png` | `plot_straight_bpm_heatmap` |
| `06_straight_bpm_input_output_profiles.py` | `06_straight_bpm_input_output_profiles.png` | `plot_straight_profile_comparison` |
| `07_parallel_coupler_heatmap.py` | `07_parallel_coupler_heatmap.png` | `plot_coupler_t3_heatmap` (alias `plot_parallel_coupler_heatmap`) |
| `08_parallel_coupler_profiles.py` | `08_parallel_coupler_profiles.png` | `plot_parallel_coupler_profiles` |
| `09_parallel_coupler_power.py` | `09_parallel_coupler_power.png` | `plot_parallel_power_transfer` |
| `10_coupler_heatmap.py` | `10_coupler_heatmap.png` | `plot_coupler_heatmap` |
| `11_coupler_port_power.py` | `11_coupler_port_power.png` | `plot_coupler_power_transfer` |
| `12_coupler_input_output_profiles.py` | `12_coupler_input_output_profiles.png` | `plot_coupler_output_profiles` (wrapper over `plot_input_output_profiles`) |
| `13_coupler_power_and_profiles.py` | `13_coupler_power_and_profiles.png` | `plot_coupler_transfer_and_profiles` |
| `14_coupler_spectral_response.py` | `14_coupler_spectral_response.png` | `plot_coupler_wavelength_response` |
| `15_sweep_parameter_to_lc.py` | `15_sweep_parameter_to_lc.png` | `plot_parameter_to_lc` |
| `16_sweep_parameter_lc.py` | `16_sweep_parameter_lc.png` | `plot_parameter_lc` |
| `17_sweep_parameter_peak_transfer.py` | `17_sweep_parameter_peak_transfer.png` | `plot_parameter_peak_transfer` |
| `18_sweep_gap_to_lc_coupler.py` | `18_sweep_gap_to_lc_coupler.png` | `plot_gap_to_lc_coupler` |
| `19_sweep_gap_to_peak_transfer_coupler.py` | `19_sweep_gap_to_peak_transfer_coupler.png` | `plot_gap_to_peak_transfer_coupler` |
| `20_sweep_gap_parallel_lc_combo.py` | `20_sweep_gap_parallel_lc_combo.png` | `plot_gap_to_lc_parallel_coupler` |
| `21_sweep_gap_parallel_lc_curve.py` | `21_sweep_gap_parallel_lc_curve.png` | `plot_gap_to_lc_curve_parallel_coupler` |
| `22_sweep_gap_parallel_peak_transfer.py` | `22_sweep_gap_parallel_peak_transfer.png` | `plot_gap_to_peak_transfer_parallel_coupler` |

Not given as separate scripts (same figure layout as above): `plot_coupler_t3_profiles_and_transfer` / `plot_parallel_coupler_profiles_and_transfer` (two figures: run `08` and `09`); `plot_gap_to_lc_curve_coupler` (same style as `plot_parameter_lc` / script `16` with coupler-oriented labels).

## Package defaults (library)

These are the **default field values** if you construct the dataclasses without passing arguments (sources: `WaveguideModalBPM1D/parameters.py`, `WaveguideModalBPM1D/analyzer.py`). They are **not** what the example gallery uses; the gallery values are defined in `examples/example_parameters.py`.

### `ModalStructureParams`

| Field | Default |
|-------|---------|
| `lambda_0` | `1.0 * um` |
| `n_points` | `1000` |
| `refractive_indices` | `(3.55, 3.55, 3.60, 3.55, 3.55)` |
| `thicknesses` | `(5.0*um, 0.5*um, 1.0*um, 0.5*um, 5.0*um)` |

### `NotebookConfig`

| Field | Default |
|-------|---------|
| `lambda_0` | `1.0 * um` |
| `n_points` | `1000` |
| `refractive_indices` | `(3.55, 3.60, 3.55)` |
| `thicknesses` | `(50.0*um, 1.0*um, 50.0*um)` |
| `joblib_n_jobs` | `-1` |
| `joblib_backend` | `"loky"` |
| `joblib_verbose` | `5` |

### `CouplerTestParams`

| Field | Default |
|-------|---------|
| `n_core` | `1.50` |
| `n_clad` | `1.42` |
| `window` | `40.0 * um` |
| `lambda_launch` | `1.0 * um` |
| `length_total` | `1764.0 * um` |
| `length_coupling` | `882.0 * um` |
| `port_offset` | `10.0 * um` |
| `core_width` | `1.0 * um` |
| `center_offset` | `1.0 * um` |
| `sigma` | `0.45 * um` |
| `n_points` | `401` |
| `mode_search_count` | `8` |

### Other API defaults (selected)

`WaveguideModalBPM1D.solve_modes`: `polarization="TE"`, `thicknesses=None` (uses config), `n_points=None`, `lambda_launch=None`, `indices=None`, `n_cladding=None` (then cladding index is taken from the outer layers of `indices`), `k_search=10`.  
`WaveguideModalBPM1D.compute_dispersion`: `polarization="TE"`, `n_x=1000`, `max_modes_cap=10`.  
`sweep_gap_to_lc_coupler` / `sweep_gap_to_lc_parallel_coupler`: `min_fraction=0.05`, `min_power=0.05` (and parallel sweep `extra_margin=5.0*um`, default `gap_values` if omitted).  
`sweep_coupler_wavelength_response`: `n_jobs=-1`, `backend="loky"`, `verbose=0`.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

## Local installation

```bash
pip install -e .
```

To install development dependencies:

```bash
pip install -e ".[dev]"
```

To enable optional acceleration with `numba`:

```bash
pip install -e ".[accel]"
```

## Quick example

```python
from WaveguideModalBPM1D import NotebookConfig, WaveguideModalBPM1D, um

config = NotebookConfig(lambda_0=1.0 * um, n_points=400)
analyzer = WaveguideModalBPM1D(config)
modal_te = analyzer.solve_modes(polarization="TE")

print(modal_te["n_effs"])
```

## Tests

```bash
pytest
```
