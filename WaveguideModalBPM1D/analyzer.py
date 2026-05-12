from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from .bpm_tridiagonal import (
    crank_nicolson_tridiagonal_coefficients,
    pack_tridiagonal_banded,
    solve_banded_tridiagonal,
    tridiagonal_matvec,
)
from .constants import nm, um

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - fallback for minimal environments
    Parallel = None
    delayed = None


@dataclass(frozen=True)
class NotebookConfig:
    lambda_0: float = 1.0 * um
    n_points: int = 1000
    refractive_indices: tuple[float, ...] = (3.55, 3.60, 3.55)
    thicknesses: tuple[float, ...] = (50.0 * um, 1.0 * um, 50.0 * um)
    joblib_n_jobs: int = -1
    joblib_backend: str = "loky"
    joblib_verbose: int = 5
    boundary_condition: str = "dirichlet"


class WaveguideModalBPM1D:
    def __init__(self, config: NotebookConfig):
        self.config = config
        self.nm = nm
        self.um = um
        self.lambda_0 = config.lambda_0
        self.k0 = 2 * np.pi / self.lambda_0
        self.n_points = config.n_points
        self.refractive_indices = np.asarray(config.refractive_indices, dtype=float)
        self.thicknesses = np.asarray(config.thicknesses, dtype=float)
        self.joblib_n_jobs = config.joblib_n_jobs
        self.joblib_backend = config.joblib_backend
        self.joblib_verbose = config.joblib_verbose
        self.boundary_condition = config.boundary_condition.lower()

    def _estimate_tbc_gamma(self, n_profile: np.ndarray, n_eff_ref: float, k0_value: float) -> complex:
        n_edge = 0.5 * (float(n_profile[0]) + float(n_profile[-1]))
        return complex(np.sqrt((k0_value * n_eff_ref) ** 2 - (k0_value * n_edge) ** 2 + 0j))

    def index_profile(self, x_vec, indices, boundaries):
        conditions = [x_vec <= boundaries[0]]
        for i in range(len(boundaries) - 1):
            conditions.append((x_vec > boundaries[i]) & (x_vec <= boundaries[i + 1]))
        return np.piecewise(x_vec, conditions, indices)

    def build_multilayer_geometry(self, thicknesses=None, n_points=None):
        thicknesses = np.asarray(
            self.thicknesses if thicknesses is None else thicknesses,
            dtype=float,
        )
        n_points = self.n_points if n_points is None else int(n_points)
        width = float(np.sum(thicknesses))
        boundaries = np.cumsum(thicknesses) - width / 2
        x_vec = np.linspace(-width / 2, width / 2, n_points)
        dx = width / (n_points - 1)
        n_profile = self.index_profile(x_vec, self.refractive_indices, boundaries)
        return {
            "thicknesses": thicknesses,
            "W": width,
            "boundaries": boundaries,
            "x": x_vec,
            "dx": dx,
            "n_profile": n_profile,
        }

    def assemble_helmholtz_operator(self, n_profile, dx, k0_value=None, polarization="TE"):
        k0_value = self.k0 if k0_value is None else k0_value
        n_points = len(n_profile)
        t1 = 1.0 / dx**2

        laplacian = diags(
            [np.ones(n_points - 1), np.full(n_points, -2.0), np.ones(n_points - 1)],
            [-1, 0, 1],
            format="csr",
        )
        identity = diags([np.ones(n_points)], [0], format="csr")

        if polarization.upper() == "TE":
            operator_a = (t1 / k0_value**2) * laplacian + diags(
                [n_profile**2], [0], format="csr"
            )
            operator_b = identity
        else:
            inv_n2_diag = diags([1.0 / n_profile**2], [0], format="csr")
            operator_a = (inv_n2_diag @ (t1 / k0_value**2 * laplacian)) + identity
            operator_b = inv_n2_diag

        return operator_a, operator_b

    def solve_guided_modes(self, operator_a, operator_b, n_core, n_cladding, k_search=10):
        eigenvalues, eigenvectors = eigsh(
            operator_a,
            k=k_search,
            M=operator_b,
            sigma=n_core**2,
            which="LM",
        )
        mask = (eigenvalues > n_cladding**2) & (eigenvalues < n_core**2 + 0.01)
        if not np.any(mask):
            return np.array([]), np.zeros((operator_a.shape[0], 0))

        neff2 = eigenvalues[mask].real
        order = np.argsort(neff2)[::-1]
        return np.sqrt(neff2[order]), eigenvectors[:, mask][:, order]

    def solve_modes(
        self,
        polarization="TE",
        thicknesses=None,
        n_points=None,
        lambda_launch=None,
        indices=None,
        n_cladding=None,
        k_search=10,
    ):
        k0_value = self.k0 if lambda_launch is None else 2 * np.pi / lambda_launch
        indices = self.refractive_indices if indices is None else np.asarray(indices, dtype=float)
        geometry = self.build_multilayer_geometry(
            thicknesses=self.thicknesses if thicknesses is None else thicknesses,
            n_points=n_points,
        )
        geometry["n_profile"] = self.index_profile(geometry["x"], indices, geometry["boundaries"])

        operator_a, operator_b = self.assemble_helmholtz_operator(
            geometry["n_profile"], geometry["dx"], k0_value=k0_value, polarization=polarization
        )
        n_core = float(np.max(indices))
        if n_cladding is None:
            n_cladding = float(max(indices[0], indices[-1]))

        n_effs, modes = self.solve_guided_modes(
            operator_a,
            operator_b,
            n_core=n_core,
            n_cladding=n_cladding,
            k_search=min(k_search, len(geometry["x"]) - 2),
        )
        return {
            **geometry,
            "A": operator_a,
            "B": operator_b,
            "n_effs": n_effs,
            "modes": modes,
            "N_modes": len(n_effs),
            "k0": k0_value,
            "polarization": polarization.upper(),
            "indices": np.asarray(indices, dtype=float),
        }

    def compute_dispersion(self, d_norm_vec, polarization="TE", n_x=1000, max_modes_cap=10):
        core_idx = int(np.argmax(self.refractive_indices))
        n_core = float(self.refractive_indices[core_idx])
        n_clad_edge = float(min(self.refractive_indices[0], self.refractive_indices[-1]))
        na_value = np.sqrt(n_core**2 - n_clad_edge**2)
        v_max = self.k0 * (d_norm_vec[-1] * self.lambda_0 / 2) * na_value
        max_modes = min(int(np.ceil(2 * v_max / np.pi)) + 1, max_modes_cap)

        def resolve_single_point(d_norm):
            local_thicknesses = self.thicknesses.copy()
            local_thicknesses[core_idx] = d_norm * self.lambda_0
            column = np.full(max_modes, np.nan)
            try:
                result = self.solve_modes(
                    polarization=polarization,
                    thicknesses=local_thicknesses,
                    n_points=n_x,
                    n_cladding=n_clad_edge,
                    k_search=min(max_modes + 2, n_x - 2),
                )
                n_effs_local = result["n_effs"]
                for mode_idx in range(min(len(n_effs_local), max_modes)):
                    column[mode_idx] = n_effs_local[mode_idx]
            except Exception:
                pass
            return column

        if Parallel is None or delayed is None:
            results = [resolve_single_point(d_norm) for d_norm in d_norm_vec]
        else:
            results = Parallel(
                n_jobs=self.joblib_n_jobs,
                backend=self.joblib_backend,
                verbose=self.joblib_verbose,
                batch_size="auto",
            )(
                delayed(resolve_single_point)(d_norm) for d_norm in d_norm_vec
            )
        return {
            "d_norm_vec": d_norm_vec,
            "neff_matrix": np.array(results).T,
            "max_modes": max_modes,
            "core_idx": core_idx,
            "n_core": n_core,
            "n_clad_edge": n_clad_edge,
            "NA": na_value,
            "V_max": v_max,
        }

    def create_bpm_context(self, mode_result, length_total):
        n_ref = float(mode_result["n_effs"][0])
        dz = self.lambda_0 / (4 * n_ref)
        nz = int(length_total / dz)
        x_vec = mode_result["x"]
        dx = mode_result["dx"]
        n_points = len(x_vec)
        return {
            "L_total": length_total,
            "dz": dz,
            "nz": nz,
            "alpha1": -1j / (2 * mode_result["k0"] * n_ref),
            "alpha2": -1j * mode_result["k0"] / (2 * n_ref),
            "I": diags([np.ones(n_points)], [0], format="csr"),
            "laplacian": diags(
                [np.ones(n_points - 1), np.full(n_points, -2.0), np.ones(n_points - 1)],
                [-1, 0, 1],
                format="csr",
            ),
            "T1": 1.0 / dx**2,
            "n_ref": n_ref,
        }

    def run_bpm_propagation(
        self,
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
        boundary_condition=None,
    ):
        _ = laplacian, identity  # assinatura mantida para compatibilidade com create_bpm_context
        field_history = np.zeros((nz, n_points), dtype=complex)
        field_history[0, :] = initial_field
        boundary_condition = (
            self.boundary_condition if boundary_condition is None else boundary_condition.lower()
        )
        tbc_gamma = None
        if boundary_condition == "tbc":
            n_eff_ref = float(n_effs[0])
            k0_value = 2j * n_eff_ref * alpha2
            tbc_gamma = self._estimate_tbc_gamma(
                n_profile,
                n_eff_ref=n_eff_ref,
                k0_value=float(np.real(k0_value)),
            )

        lo_a, mid_a, up_a, lo_b, mid_b, up_b = crank_nicolson_tridiagonal_coefficients(
            alpha1,
            alpha2,
            t1,
            dz,
            n_profile,
            float(n_effs[0]),
            boundary_condition=boundary_condition,
            tbc_gamma=tbc_gamma,
        )
        ab_a = pack_tridiagonal_banded(lo_a, mid_a, up_a)

        print(f"Propagando por {nz} passos...")
        for i in range(1, nz):
            rhs = tridiagonal_matvec(lo_b, mid_b, up_b, field_history[i - 1, :])
            field_history[i, :] = solve_banded_tridiagonal(ab_a, rhs)

        return field_history

    def generate_gaussian_input(self, x_vec, xc=0.0, sigma=None, amplitude=1.0):
        sigma = 1.0 * self.um if sigma is None else sigma
        return amplitude * np.exp(-((x_vec - xc) / sigma) ** 2)

    def normalize_field(self, x_vec, field):
        power = trapezoid(np.abs(field) ** 2, x_vec)
        return field / np.sqrt(power)

    def build_dual_core_profile(self, x_vec, center_offset, core_width, n_core, n_clad):
        n_prof = np.full_like(x_vec, n_clad, dtype=float)
        upper_core = np.abs(x_vec - center_offset) <= (core_width / 2)
        lower_core = np.abs(x_vec + center_offset) <= (core_width / 2)
        n_prof[upper_core | lower_core] = n_core
        return n_prof

    def power_in_mask(self, x_vec, field, mask):
        return trapezoid(np.abs(field[mask]) ** 2, x_vec[mask])

    def estimate_coupling_length(self, z_vec, power_lower_norm, min_fraction=0.05):
        delta_power = np.diff(power_lower_norm)
        peak_candidates = np.where((delta_power[:-1] > 0) & (delta_power[1:] <= 0))[0] + 1
        min_start_idx = max(5, int(min_fraction * len(z_vec)))
        valid_peaks = [
            idx for idx in peak_candidates if idx >= min_start_idx and power_lower_norm[idx] > 0.05
        ]
        if valid_peaks:
            lc_idx = valid_peaks[0]
        else:
            lc_idx = min_start_idx + np.argmax(power_lower_norm[min_start_idx:])
        return lc_idx, z_vec[lc_idx]

    def build_bent_coupler_profile(self, x_vec, offset, core_width, n_core, n_clad):
        return self.build_dual_core_profile(
            x_vec=x_vec,
            center_offset=offset,
            core_width=core_width,
            n_core=n_core,
            n_clad=n_clad,
        )

    def run_z_variant_bpm(
        self,
        initial_field,
        x_vec,
        z_vec,
        n_ref,
        k0_value,
        profile_builder,
        edge_signature,
        boundary_condition=None,
    ):
        nx_local = len(x_vec)
        dx_local = x_vec[1] - x_vec[0]
        dz_local = z_vec[1] - z_vec[0]
        alpha1_local = -1j / (2 * k0_value * n_ref)
        alpha2_local = -1j * k0_value / (2 * n_ref)
        t1_local = 1.0 / dx_local**2

        field_history = np.zeros((len(z_vec), nx_local), dtype=complex)
        field_history[0, :] = initial_field
        operator_cache = {}
        boundary_condition = (
            self.boundary_condition if boundary_condition is None else boundary_condition.lower()
        )

        for iz in range(1, len(z_vec)):
            z_mid = 0.5 * (z_vec[iz] + z_vec[iz - 1])
            cache_key = edge_signature(z_mid)

            if cache_key not in operator_cache:
                n_prof_mid = profile_builder(z_mid)
                tbc_gamma = None
                if boundary_condition == "tbc":
                    tbc_gamma = self._estimate_tbc_gamma(
                        n_prof_mid,
                        n_eff_ref=n_ref,
                        k0_value=k0_value,
                    )
                lo_a, mid_a, up_a, lo_b, mid_b, up_b = crank_nicolson_tridiagonal_coefficients(
                    alpha1_local,
                    alpha2_local,
                    t1_local,
                    dz_local,
                    n_prof_mid,
                    n_ref,
                    boundary_condition=boundary_condition,
                    tbc_gamma=tbc_gamma,
                )
                ab_a = pack_tridiagonal_banded(lo_a, mid_a, up_a)
                operator_cache[cache_key] = (ab_a, lo_b, mid_b, up_b)

            ab_a, lo_b, mid_b, up_b = operator_cache[cache_key]
            rhs = tridiagonal_matvec(lo_b, mid_b, up_b, field_history[iz - 1, :])
            field_history[iz, :] = solve_banded_tridiagonal(ab_a, rhs)

        return field_history


WaveguideAnalysis2D = WaveguideModalBPM1D
