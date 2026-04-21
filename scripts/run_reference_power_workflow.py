#!/usr/bin/env python3

from __future__ import annotations

# ------------------
# Imports
# ------------------
# general -----
import json
import sys
from pathlib import Path

# scientific -----
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# local -----
from fit.fit_eft import (
    DEFAULT_K_REF_H_PER_MPC,
    DEFAULT_MAX_ABS_DRIFT_SIGMA,
    DEFAULT_MAX_CHI2_DOF,
    DEFAULT_MAX_HOLDOUT_CHI2_DOF,
    fit_counterterm,
    load_measurement_artifact,
    load_theory_artifact,
    run_kmax_scan,
    select_k_stability,
)
from fit.plotting import save_all_power_figures, save_main_spectrum_comparison, save_redshift_stability_summary
from theory.header_cosmo import cosmo_from_header
from theory.linear_baseline import make_linear_pk
from theory.spt_one_loop import compute_one_loop
from theory.validation import validate_one_loop_theory

# ------------------
# Constants
# ------------------
DEFAULT_SCAN_VALUES = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
UNUSED_FIT_ARTIFACTS = [
    "analysis_scope_notes.json",
    "comparison_requested_fit.png",
    "eft_fit_residuals.csv",
    "figure_manifest.json",
    "gaussian_information_curve.csv",
    "gaussian_information_curve.metadata.json",
    "kmax_scan.metadata.json",
    "one_loop_validation_curve.csv",
    "one_loop_validation_summary.json",
    "requested_fit_summary.json",
    "fit_summary.json",
]

# ------------------
# Functions
# ------------------
def _json(data: dict, path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)),
        encoding="utf-8",
    )


def estimate_k_nl(k: np.ndarray, pk: np.ndarray) -> float | None:
    """Interpolate the k where Delta^2(k) = 1.

    Args:
        k: wavenumbers in h/Mpc
        pk: linear power in (Mpc/h)^3
    """
    delta2 = k**3 * pk / (2.0 * np.pi**2)
    if not np.any(delta2 >= 1.0):
        return None
    idx = int(np.argmax(delta2 >= 1.0))
    if idx == 0:
        return float(k[0])
    x0, x1 = np.log(k[idx - 1]), np.log(k[idx])
    y0, y1 = np.log(delta2[idx - 1]), np.log(delta2[idx])
    return float(np.exp(x0 + (0.0 - y0) / (y1 - y0) * (x1 - x0)))


def run_single_redshift(
    *,
    z_choice: float,
    fit_kmax: float,
    nr: int,
    nx: int,
    k_ref_h_per_Mpc: float,
    max_chi2_dof: float,
    max_abs_drift_sigma: float,
    max_holdout_chi2_dof: float,
    extended_figures: bool,
) -> dict:
    source_csv = ROOT / "data" / f"z{z_choice:.3f}" / "AB" / "power_nfft2048.csv"
    header_path = ROOT / "data" / f"z{z_choice:.3f}" / "AB" / "header"
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing source power spectrum {source_csv}.")

    measurement = load_measurement_artifact(source_csv)
    cosmo = cosmo_from_header(header_path)

    measurement_dir = ROOT / "artifacts" / "measurements"
    theory_dir = ROOT / "artifacts" / "theory" / f"z{z_choice:.3f}"
    fit_dir = ROOT / "artifacts" / "fit" / f"z{z_choice:.3f}"
    for d in (measurement_dir, theory_dir, fit_dir):
        d.mkdir(parents=True, exist_ok=True)

    # keep the fit folder from collecting stale products
    for name in UNUSED_FIT_ARTIFACTS:
        path = fit_dir / name
        if path.exists():
            path.unlink()

    # write measurement
    meas_cols = {"k_h_per_Mpc": measurement["k_h_per_Mpc"], "P_meas": measurement["P_meas"]}
    if "nmodes" in measurement:
        meas_cols["nmodes"] = measurement["nmodes"]
    pd.DataFrame(meas_cols).to_csv(measurement_dir / f"matter_pk_real_z{z_choice:.3f}.csv", index=False)

    # build and write linear theory
    k_lin = np.geomspace(2.0 * np.pi / cosmo["box_size"] * 0.5, 1.0, 300)
    pk_lin = make_linear_pk(k_lin, cosmo)
    pd.DataFrame({"k_h_per_Mpc": k_lin, "P_lin_Mpc3_per_h3": pk_lin}).to_csv(
        theory_dir / f"pk_linear_z{z_choice:.3f}.csv", index=False
    )

    # compute and write one-loop theory
    theory = compute_one_loop(measurement["k_h_per_Mpc"], k_lin, pk_lin, nr=nr, nx=nx)
    theory_csv = theory_dir / "pk_spt_one_loop_for_fit.csv"
    pd.DataFrame({
        "k_h_per_Mpc": theory["k"],
        "P11": theory["P11"],
        "P22": theory["P22"],
        "P13": theory["P13"],
        "P_1loop": theory["P_1loop"],
    }).to_csv(theory_csv, index=False)

    theory_loaded = load_theory_artifact(theory_csv)
    validation = validate_one_loop_theory(
        theory_columns=theory_loaded,
        linear_k_h_per_Mpc=k_lin,
        linear_power_Mpc3_per_h3=pk_lin,
    )

    # fit at k_max
    fit = fit_counterterm(measurement, theory_loaded, k_max_h_per_Mpc=fit_kmax)

    # k_max stability scan
    scan_values = sorted({*DEFAULT_SCAN_VALUES, float(fit_kmax), float(k_ref_h_per_Mpc)})
    scan = run_kmax_scan(
        measurement=measurement,
        theory=theory_loaded,
        k_max_values_h_per_Mpc=scan_values,
        k_ref_h_per_Mpc=k_ref_h_per_Mpc,
    )
    selection = select_k_stability(
        scan=scan,
        max_chi2_dof=max_chi2_dof,
        max_abs_drift_sigma=max_abs_drift_sigma,
        max_holdout_chi2_dof=max_holdout_chi2_dof,
    )

    selected_k = float(selection["selected_k_stab_h_per_Mpc"])
    selected_index = int(np.argmin(np.abs(np.array([f["k_max_h_per_Mpc"] for f in scan["fits"]]) - selected_k)))
    selected_fit = scan["fits"][selected_index]
    selected_row = selection["selected_row"]

    # write scan + selected fit curves
    pd.DataFrame(scan["rows"]).to_csv(fit_dir / "kmax_scan.csv", index=False)
    pd.DataFrame(selected_fit["fit_columns"]).to_csv(fit_dir / "eft_fit_curve.csv", index=False)

    selected_summary = {k: v for k, v in selected_fit.items() if k not in {"fit_columns", "residual_columns"}}
    selected_summary.update({
        "selected_k_stab_h_per_Mpc": selected_k,
        "parameter_drift_sigma": float(selected_row["parameter_drift_sigma"]),
        "chi2_holdout_dof": float(selected_row["chi2_holdout_dof"]),
        "selection_status": selection["selection_status"],
        "selection_note": selection["selection_note"],
        "selected_row_passes_all_cuts": selection["selected_row_passes_all_cuts"],
    })
    _json(selected_summary, fit_dir / "eft_fit_summary.json")
    _json({
        "redshift": z_choice,
        "selected_k_stab_h_per_Mpc": selected_k,
        "c_counterterm_Mpch2": selected_fit["c_counterterm_Mpch2"],
        "sigma_c_counterterm_Mpch2": selected_fit["sigma_c_counterterm_Mpch2"],
        "chi2_dof": selected_fit["chi2_dof"],
        "chi2_holdout_dof": float(selected_row["chi2_holdout_dof"]),
        "parameter_drift_sigma": float(selected_row["parameter_drift_sigma"]),
        "selection_status": selection["selection_status"],
        "selected_row_passes_all_cuts": selection["selected_row_passes_all_cuts"],
    }, fit_dir / "breakdown_scale_summary.json")

    # figures
    save_all_power_figures(
        output_dir=fit_dir,
        fit=selected_fit,
        scan=scan,
        scale_markers=None,
        selected_k_stab=None,
        theory_validation=validation,
        include_extended=extended_figures,
    )
    save_main_spectrum_comparison(
        k=fit["fit_columns"]["k_h_per_Mpc"],
        p_meas=fit["fit_columns"]["P_meas"],
        p11=fit["fit_columns"]["P11"],
        p_1loop=fit["fit_columns"]["P_1loop"],
        p_eft=fit["fit_columns"]["P_eft"],
        output_path=fit_dir / "comparison_fit.png",
        scale_markers=None,
        residual_yscale="linear",
        residual_ylim=(-0.2, 0.2),
    )

    return {
        "redshift": float(z_choice),
        "fit_dir": fit_dir,
        "selected_k_stab_h_per_Mpc": selected_k,
        "c_counterterm_Mpch2": float(selected_fit["c_counterterm_Mpch2"]),
        "sigma_c_counterterm_Mpch2": float(selected_fit["sigma_c_counterterm_Mpch2"]),
        "chi2_dof": float(selected_fit["chi2_dof"]),
        "chi2_holdout_dof": float(selected_row["chi2_holdout_dof"]),
        "parameter_drift_sigma": float(selected_row["parameter_drift_sigma"]),
        "fit_kmax_h_per_Mpc": float(fit_kmax),
        "k_ref_h_per_Mpc": float(scan["metadata"]["k_ref_h_per_Mpc"]),
        "selection_status": selection["selection_status"],
        "selected_row_passes_all_cuts": selection["selected_row_passes_all_cuts"],
    }


# ------------------
# Parameters
# ------------------
Z_VALUES      = [0.5]
FIT_KMAX      = 0.20
NR            = 256
NX            = 64
K_REF         = DEFAULT_K_REF_H_PER_MPC
MAX_CHI2_DOF  = DEFAULT_MAX_CHI2_DOF
MAX_DRIFT     = DEFAULT_MAX_ABS_DRIFT_SIGMA
MAX_HOLD_CHI2 = DEFAULT_MAX_HOLDOUT_CHI2_DOF
EXTENDED_FIGS = False

# ------------------
# Run
# ------------------
suite_rows = [
    run_single_redshift(
        z_choice=z,
        fit_kmax=FIT_KMAX,
        nr=NR,
        nx=NX,
        k_ref_h_per_Mpc=K_REF,
        max_chi2_dof=MAX_CHI2_DOF,
        max_abs_drift_sigma=MAX_DRIFT,
        max_holdout_chi2_dof=MAX_HOLD_CHI2,
        extended_figures=EXTENDED_FIGS,
    )
    for z in Z_VALUES
]

if len(suite_rows) > 1:
    suite_dir = ROOT / "artifacts" / "fit" / "redshift_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(suite_rows).to_csv(suite_dir / "redshift_breakdown_summary.csv", index=False)
    save_redshift_stability_summary(
        summary_rows=suite_rows,
        output_path=suite_dir / "redshift_breakdown_summary.png",
    )
