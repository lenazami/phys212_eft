from __future__ import annotations

# ------------------
# Imports
# ------------------

# general -----
import json
import re
from pathlib import Path

# numpy -----
import numpy as np

# local -----
from theory.eft_model import evaluate_eft_model
from theory.header_cosmo import cosmo_from_header, parse_header

# ------------------
# Constants
# ------------------

PROJECT_UNITS = {
    "k_h_per_Mpc": "h/Mpc",
    "power_spectrum": "(Mpc/h)^3",
    "c_counterterm_Mpch2": "(Mpc/h)^2",
}

ONE_LOOP_CONVENTIONS = {
    "observable_type": "real-space matter power spectrum",
    "P11_definition": "linear matter power spectrum",
    "P22_definition": "2 * integral[F2^2 * P11(q) * P11(|k-q|)]",
    "P13_definition": "the full 6*F3 contribution, so the repo uses P_1loop = P11 + P22 + P13",
    "P_1loop_definition": "P11 + P22 + P13",
    "k_units": "h/Mpc",
    "power_units": "(Mpc/h)^3",
}

EFT_CONVENTIONS = {
    "counterterm_definition": "P_ctr = -2 c_counterterm k^2 P11",
    "counterterm_sign_note": "positive c_counterterm suppresses power in this convention",
    "counterterm_units": "(Mpc/h)^2",
}

REPORT_CONVENTIONS = [
    "The repo convention is P_1loop = P11 + P22 + P13, with P13 already including the full mirror contribution.",
    "The EFT counterterm is P_ctr = -2 c_counterterm k^2 P11, so the sign of c_counterterm must be interpreted with that minus sign included.",
    "All live power-spectrum arrays use k in h/Mpc and P(k) in (Mpc/h)^3.",
    "The linear baseline is interpolated from the CLASS power spectrum stored in official Abacus metadata and scaled to the target epoch with the Abacus growth table.",
]
SHARED_CONVENTIONS = {**ONE_LOOP_CONVENTIONS, **EFT_CONVENTIONS}

DEFAULT_K_REF_H_PER_MPC = 0.10
DEFAULT_MAX_CHI2_DOF = 1.5
DEFAULT_MAX_ABS_DRIFT_SIGMA = 1.0
DEFAULT_MAX_HOLDOUT_CHI2_DOF = 10.0
DEFAULT_HOLDOUT_WIDTH_H_PER_MPC = 0.04

# ------------------
# Functions
# ------------------


def load_measurement_artifact(csv_path: Path, metadata_path: Path | None = None) -> dict:
    lines = [
        line.strip()
        for line in csv_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    header = lines[0]
    names = [name.strip() for name in header.split(",")] if "," in header else header.split()
    delimiter = "," if "," in header else None
    data = np.genfromtxt(lines[1:], delimiter=delimiter, names=names, dtype=np.float64)
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)
    cols = {name: np.asarray(data[name], dtype=np.float64) for name in data.dtype.names}

    k_name = next((name for name in ("k_h_per_Mpc", "kavg", "k", "k_center") if name in cols), "kavg")
    power_name = next((name for name in ("P0", "P_meas", "power", "P") if name in cols), "P0")

    k = np.asarray(cols[k_name], dtype=np.float64)
    p_meas = np.asarray(cols[power_name], dtype=np.float64)
    keep = np.isfinite(k) & np.isfinite(p_meas) & (k > 0.0)

    out = {
        "k_h_per_Mpc": k,
        "P_meas": p_meas,
    }
    if "nmodes" in cols:
        nmodes = np.asarray(cols["nmodes"], dtype=np.float64)
        keep &= np.isfinite(nmodes) & (nmodes > 0.0)
        out["nmodes"] = nmodes
    elif "N_modes" in cols:
        nmodes = np.asarray(cols["N_modes"], dtype=np.float64)
        keep &= np.isfinite(nmodes) & (nmodes > 0.0)
        out["nmodes"] = nmodes

    out = {name: values[keep] for name, values in out.items()}
    order = np.argsort(out["k_h_per_Mpc"])
    out = {name: values[order] for name, values in out.items()}

    meta = {"source_power_csv": str(csv_path)}
    header_path = csv_path.parent / "header"
    if header_path.exists():
        meta["source_header"] = str(header_path)
        try:
            raw = parse_header(header_path)
        except OSError:
            raw = {}
        if "SimName" in raw:
            meta["sim_name"] = str(raw["SimName"])
        if "Redshift" in raw:
            meta["redshift"] = float(raw["Redshift"])
        if "BoxSizeHMpc" in raw:
            meta["box_size_hMpc"] = float(raw["BoxSizeHMpc"])

    if "redshift" not in meta:
        match = re.search(r"/z([0-9]+(?:\.[0-9]+)?)/", str(csv_path))
        if match is not None:
            meta["redshift"] = float(match.group(1))

    match = re.search(r"nfft([0-9]+)", csv_path.stem)
    if match is not None:
        meta["n_mesh"] = int(match.group(1))
        box_size = meta.get("box_size_hMpc")
        if box_size is not None:
            meta["k_nyquist_h_per_Mpc"] = float(np.pi * meta["n_mesh"] / box_size)

    if metadata_path is None:
        metadata_path = csv_path.with_suffix(".metadata.json")
    if metadata_path.exists():
        meta.update(json.loads(metadata_path.read_text(encoding="utf-8")))

    meta.setdefault("artifact_path", str(csv_path))
    meta.setdefault("observable_type", "matter_pk_real")
    meta.setdefault("los_choice", "none")
    meta.setdefault("k_column", k_name)
    meta.setdefault("power_column", power_name)
    meta.setdefault("weighting_note", "Use 2 P(k)^2 / N_modes when nmodes are present, else uniform weights.")
    meta.setdefault("units", dict(PROJECT_UNITS))
    meta["nmodes_present"] = bool("nmodes" in out)
    out["metadata"] = meta
    return out


def load_theory_artifact(csv_path: Path, metadata_path: Path | None = None) -> dict:
    lines = [
        line.strip()
        for line in csv_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    header = lines[0]
    names = [name.strip() for name in header.split(",")] if "," in header else header.split()
    delimiter = "," if "," in header else None
    data = np.genfromtxt(lines[1:], delimiter=delimiter, names=names, dtype=np.float64)
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)
    cols = {name: np.asarray(data[name], dtype=np.float64) for name in data.dtype.names}

    out = {
        "k_h_per_Mpc": np.asarray(cols["k_h_per_Mpc"], dtype=np.float64),
        "P11": np.asarray(cols["P11"], dtype=np.float64),
        "P22": np.asarray(cols["P22"], dtype=np.float64),
        "P13": np.asarray(cols["P13"], dtype=np.float64),
        "P_1loop": np.asarray(cols["P_1loop"], dtype=np.float64) if "P_1loop" in cols else np.asarray(cols["P11"] + cols["P22"] + cols["P13"], dtype=np.float64),
    }
    keep = np.isfinite(out["k_h_per_Mpc"]) & (out["k_h_per_Mpc"] > 0.0)
    for name in ("P11", "P22", "P13", "P_1loop"):
        keep &= np.isfinite(out[name])
    out = {name: values[keep] for name, values in out.items()}
    order = np.argsort(out["k_h_per_Mpc"])
    out = {name: values[order] for name, values in out.items()}

    meta = {}
    if metadata_path is None:
        metadata_path = csv_path.with_suffix(".metadata.json")
    if metadata_path.exists():
        meta.update(json.loads(metadata_path.read_text(encoding="utf-8")))
    meta.setdefault("artifact_path", str(csv_path))
    meta.setdefault("observable_type", "matter_pk_real")
    meta.setdefault("units", dict(PROJECT_UNITS))
    meta.setdefault("conventions", dict(SHARED_CONVENTIONS))
    meta.setdefault("report_conventions", list(REPORT_CONVENTIONS))
    if "cosmology" not in meta:
        header_path = meta.get("source_header")
        if header_path and Path(header_path).exists():
            meta["cosmology"] = cosmo_from_header(Path(header_path))
    out["metadata"] = meta
    return out


def _build_validation_summary(
    *,
    measurement: dict,
    theory: dict,
    k_fit: np.ndarray,
    p11_fit: np.ndarray,
    p22_fit: np.ndarray,
    p13_fit: np.ndarray,
    c_counterterm: float,
    residual: np.ndarray,
) -> dict[str, bool]:
    p_1loop_theory = theory["P11"] + theory["P22"] + theory["P13"]
    return {
        "measurement_k_monotonic": bool(np.all(np.diff(measurement["k_h_per_Mpc"]) > 0.0)),
        "theory_k_monotonic": bool(np.all(np.diff(theory["k_h_per_Mpc"]) > 0.0)),
        "fit_k_monotonic": bool(np.all(np.diff(k_fit) > 0.0)),
        "p11_positive": bool(np.all(p11_fit > 0.0)),
        "loop_terms_finite": bool(
            np.all(np.isfinite(p22_fit))
            and np.all(np.isfinite(p13_fit))
            and np.all(np.isfinite(p11_fit + p22_fit + p13_fit))
        ),
        "theory_p1loop_matches_sum": bool(np.allclose(theory["P_1loop"], p_1loop_theory, rtol=1e-10, atol=1e-10)),
        "fitted_counterterm_finite": bool(np.isfinite(c_counterterm)),
        "residual_length_match": bool(len(residual) == len(k_fit)),
    }


def _interpolate_theory_to_data(k: np.ndarray, theory: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p11 = np.interp(k, theory["k_h_per_Mpc"], theory["P11"])
    p22 = np.interp(k, theory["k_h_per_Mpc"], theory["P22"])
    p13 = np.interp(k, theory["k_h_per_Mpc"], theory["P13"])
    return p11, p22, p13


def _compute_fit_sigma2(
    *,
    measurement: dict,
    keep: np.ndarray,
    p_for_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    if "nmodes" in measurement:
        nmodes = np.asarray(measurement["nmodes"], dtype=np.float64)[keep]
        sigma2 = 2.0 * p_for_covariance**2 / nmodes
        weight_mode = "mode_counting"
    else:
        nmodes = None
        sigma2 = np.ones_like(p_for_covariance)
        weight_mode = "uniform"

    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
        raise ValueError("Fit variances must be finite and positive.")
    return sigma2, nmodes, weight_mode


def _evaluate_model_at_k(k: np.ndarray, theory: dict, c_counterterm: float) -> dict[str, np.ndarray]:
    p11, p22, p13 = _interpolate_theory_to_data(k, theory)
    return evaluate_eft_model(k=k, p11=p11, p22=p22, p13=p13, c=c_counterterm)


def fit_counterterm(
    measurement: dict,
    theory: dict,
    k_max_h_per_Mpc: float,
    k_ren_h_per_Mpc: float | None = None,
) -> dict:
    if not np.isfinite(k_max_h_per_Mpc) or k_max_h_per_Mpc <= 0.0:
        raise ValueError("`k_max_h_per_Mpc` must be finite and positive.")

    k_meas = np.asarray(measurement["k_h_per_Mpc"], dtype=np.float64)
    p_meas_all = np.asarray(measurement["P_meas"], dtype=np.float64)
    keep = (k_meas <= k_max_h_per_Mpc) & (k_meas >= theory["k_h_per_Mpc"][0]) & (k_meas <= theory["k_h_per_Mpc"][-1])
    if not np.any(keep):
        raise ValueError("No measurement points survive this k_max.")

    k = k_meas[keep]
    p_meas = p_meas_all[keep]
    p11, p22, p13 = _interpolate_theory_to_data(k, theory)
    p_1loop = p11 + p22 + p13

    sigma2, nmodes, weight_mode = _compute_fit_sigma2(
        measurement=measurement,
        keep=keep,
        p_for_covariance=p_meas,
    )

    # Gaussian Fisher estimate for the single counterterm parameter
    basis = -2.0 * k**2 * p11
    fisher = np.sum(basis**2 / sigma2)
    if not np.isfinite(fisher) or fisher <= 0.0:
        raise ValueError("The counterterm basis is degenerate on this fit range.")

    c_counterterm = float(np.sum((p_meas - p_1loop) * basis / sigma2) / fisher)
    model = _evaluate_model_at_k(k, theory, c_counterterm)

    residual = p_meas - model["P_eft"]
    residual_linear = p_meas - p11
    residual_spt = p_meas - p_1loop
    sigma = np.sqrt(sigma2)
    sigma_c = float(1.0 / np.sqrt(fisher))
    validation_summary = _build_validation_summary(
        measurement=measurement,
        theory=theory,
        k_fit=k,
        p11_fit=p11,
        p22_fit=p22,
        p13_fit=p13,
        c_counterterm=c_counterterm,
        residual=residual,
    )

    result = {
        "observable_type": measurement["metadata"].get("observable_type", "matter_pk_real"),
        "redshift": measurement["metadata"].get("redshift", theory["metadata"].get("redshift")),
        "units": dict(PROJECT_UNITS),
        "conventions": dict(SHARED_CONVENTIONS),
        "report_conventions": list(REPORT_CONVENTIONS),
        "validation_summary": validation_summary,
        "c_counterterm_Mpch2": c_counterterm,
        "sigma_c_counterterm_Mpch2": sigma_c,
        "fisher_c_counterterm": float(fisher),
        "k_max_h_per_Mpc": float(k_max_h_per_Mpc),
        "n_points": int(len(k)),
        "dof": int(len(k) - 1),
        "chi2": float(np.sum((residual / sigma) ** 2)),
        "chi2_linear": float(np.sum((residual_linear / sigma) ** 2)),
        "chi2_spt": float(np.sum((residual_spt / sigma) ** 2)),
        "weight_mode": weight_mode,
        "covariance_assumption": "Gaussian mode counting" if weight_mode == "mode_counting" else "uniform weights",
        "measurement_metadata": measurement["metadata"],
        "theory_metadata": theory["metadata"],
        "fit_columns": {
            "k_h_per_Mpc": k,
            "P_meas": p_meas,
            "P11": model["P11"],
            "P22": model["P22"],
            "P13": model["P13"],
            "P_1loop": model["P_1loop"],
            "P_ctr": model["P_ctr"],
            "P_eft": model["P_eft"],
        },
        "residual_columns": {
            "k_h_per_Mpc": k,
            "residual": residual,
            "fractional_residual": residual / p_meas,
            "pull": residual / sigma,
            "sigma_gaussian": sigma,
        },
    }
    if nmodes is not None:
        result["residual_columns"]["nmodes"] = nmodes

    dof = result["dof"]
    result["chi2_dof"] = float(result["chi2"] / dof) if dof > 0 else np.nan
    result["chi2_improvement"] = float(result["chi2_spt"] - result["chi2"])

    if k_ren_h_per_Mpc is not None:
        i = int(np.argmin(np.abs(k - k_ren_h_per_Mpc)))
        result["k_ren_h_per_Mpc"] = float(k[i])
        result["c_counterterm_kren_Mpch2"] = float((p_1loop[i] - p_meas[i]) / (2.0 * k[i] ** 2 * p11[i]))

    return result


def run_holdout_test(
    *,
    measurement: dict,
    theory: dict,
    k_fit_max_h_per_Mpc: float,
    k_holdout_max_h_per_Mpc: float | None = None,
    holdout_width_h_per_Mpc: float = DEFAULT_HOLDOUT_WIDTH_H_PER_MPC,
    fit_result: dict | None = None,
) -> dict:
    fit = fit_result if fit_result is not None else fit_counterterm(
        measurement=measurement,
        theory=theory,
        k_max_h_per_Mpc=k_fit_max_h_per_Mpc,
    )

    k_meas = np.asarray(measurement["k_h_per_Mpc"], dtype=np.float64)
    p_meas = np.asarray(measurement["P_meas"], dtype=np.float64)
    if k_holdout_max_h_per_Mpc is None:
        k_hi_max = min(
            float(theory["k_h_per_Mpc"][-1]),
            float(k_fit_max_h_per_Mpc) + float(holdout_width_h_per_Mpc),
        )
    else:
        k_hi_max = min(float(k_holdout_max_h_per_Mpc), float(theory["k_h_per_Mpc"][-1]))
    holdout = (k_meas > k_fit_max_h_per_Mpc) & (k_meas <= k_hi_max)

    if not np.any(holdout):
        return {
            "holdout_available": False,
            "k_fit_max_h_per_Mpc": float(k_fit_max_h_per_Mpc),
            "k_holdout_max_h_per_Mpc": float(k_hi_max),
            "holdout_width_h_per_Mpc": float(k_hi_max - float(k_fit_max_h_per_Mpc)),
            "n_holdout": 0,
            "chi2_holdout": np.nan,
            "chi2_holdout_dof": np.nan,
            "chi2_holdout_spt": np.nan,
            "chi2_holdout_improvement": np.nan,
            "rms_fractional_residual_holdout": np.nan,
        }

    k_hold = k_meas[holdout]
    p_hold = p_meas[holdout]
    sigma2_hold, nmodes_hold, weight_mode = _compute_fit_sigma2(
        measurement=measurement,
        keep=holdout,
        p_for_covariance=p_hold,
    )
    sigma_hold = np.sqrt(sigma2_hold)
    model_hold = _evaluate_model_at_k(k_hold, theory, fit["c_counterterm_Mpch2"])
    residual_hold = p_hold - model_hold["P_eft"]
    residual_hold_spt = p_hold - model_hold["P_1loop"]
    dof_hold = int(len(k_hold))

    result = {
        "holdout_available": True,
        "k_fit_max_h_per_Mpc": float(k_fit_max_h_per_Mpc),
        "k_holdout_max_h_per_Mpc": float(k_hi_max),
        "holdout_width_h_per_Mpc": float(k_hi_max - float(k_fit_max_h_per_Mpc)),
        "n_holdout": int(len(k_hold)),
        "chi2_holdout": float(np.sum((residual_hold / sigma_hold) ** 2)),
        "chi2_holdout_dof": float(np.sum((residual_hold / sigma_hold) ** 2) / dof_hold) if dof_hold > 0 else np.nan,
        "chi2_holdout_spt": float(np.sum((residual_hold_spt / sigma_hold) ** 2)),
        "weight_mode": weight_mode,
        "rms_fractional_residual_holdout": float(np.sqrt(np.mean((residual_hold / p_hold) ** 2))),
    }
    result["chi2_holdout_improvement"] = float(result["chi2_holdout_spt"] - result["chi2_holdout"])
    if nmodes_hold is not None:
        result["nmodes_holdout_total"] = int(np.sum(nmodes_hold))
    return result


def build_gaussian_information_curve(
    *,
    measurement: dict,
    k_max_values_h_per_Mpc=None,
) -> dict:
    if "nmodes" not in measurement:
        raise ValueError("Gaussian information curves require `nmodes` in the measurement artifact.")

    k = np.asarray(measurement["k_h_per_Mpc"], dtype=np.float64)
    nmodes = np.asarray(measurement["nmodes"], dtype=np.float64)

    values = k if k_max_values_h_per_Mpc is None else np.sort(
        np.unique(np.asarray(k_max_values_h_per_Mpc, dtype=np.float64))
    )

    def _row(k_max: float) -> dict | None:
        keep = k <= k_max
        if not np.any(keep):
            return None
        nmodes_sum = float(np.sum(nmodes[keep]))
        sn2 = 0.5 * nmodes_sum
        return {
            "k_max_h_per_Mpc": k_max,
            "n_bins": int(np.sum(keep)),
            "cumulative_nmodes": nmodes_sum,
            "gaussian_sn2": sn2,
            "gaussian_sn": float(np.sqrt(sn2)),
            "gaussian_amp_fisher": sn2,
        }

    rows = [r for k_max in values if (r := _row(float(k_max))) is not None]
    return {
        "rows": rows,
        "metadata": {
            "observable_type": measurement["metadata"].get("observable_type", "matter_pk_real"),
            "redshift": measurement["metadata"].get("redshift"),
            "covariance_assumption": "Gaussian diagonal covariance from shell mode counts",
            "interpretation_note": "For an amplitude-like parameter with diagonal Gaussian covariance, the cumulative Fisher information reduces to one-half the cumulative number of modes.",
        },
    }


def select_k_stability(
    *,
    scan: dict,
    max_chi2_dof: float = DEFAULT_MAX_CHI2_DOF,
    max_abs_drift_sigma: float = DEFAULT_MAX_ABS_DRIFT_SIGMA,
    max_holdout_chi2_dof: float = DEFAULT_MAX_HOLDOUT_CHI2_DOF,
) -> dict:
    """Pick the largest k_max where chi2/dof, drift, and holdout all pass thresholds."""
    rows = scan["rows"]
    if not rows:
        raise ValueError("The scan must contain at least one row.")

    selected = None
    evaluated_rows = []
    for row in rows:
        chi2_ok = bool(np.isfinite(row["chi2_dof"]) and row["chi2_dof"] <= max_chi2_dof)
        drift_ok = bool(
            np.isfinite(row["parameter_drift_sigma"])
            and abs(row["parameter_drift_sigma"]) <= max_abs_drift_sigma
        )
        holdout_ok = bool(
            row["holdout_available"]
            and np.isfinite(row["chi2_holdout_dof"])
            and row["chi2_holdout_dof"] <= max_holdout_chi2_dof
        )
        accepted = chi2_ok and drift_ok and holdout_ok
        evaluated_row = {
            **row,
            "passes_chi2_cut": chi2_ok,
            "passes_drift_cut": drift_ok,
            "passes_holdout_cut": holdout_ok,
            "passes_all_cuts": accepted,
        }
        evaluated_rows.append(evaluated_row)
        if accepted:
            selected = evaluated_row

    used_fallback = selected is None
    if selected is None:
        selected = evaluated_rows[0]

    return {
        "selected_k_stab_h_per_Mpc": float(selected["k_max_h_per_Mpc"]),
        "selected_row": selected,
        "selected_row_passes_all_cuts": bool(selected["passes_all_cuts"]),
        "selection_status": "fallback_to_smallest_scale" if used_fallback else "largest_passing_scale",
        "selection_rule": {
            "max_chi2_dof": float(max_chi2_dof),
            "max_abs_drift_sigma": float(max_abs_drift_sigma),
            "max_holdout_chi2_dof": float(max_holdout_chi2_dof),
        },
        "selection_note": (
            "k_stab is taken to be the largest scanned k_max whose cumulative chi2/dof, "
            "parameter drift relative to k_ref, and local holdout chi2/dof all remain within "
            "the chosen conservative thresholds."
        ),
        "evaluated_rows": evaluated_rows,
    }


def run_kmax_scan(
    *,
    measurement: dict,
    theory: dict,
    k_max_values_h_per_Mpc,
    k_holdout_max_h_per_Mpc: float | None = None,
    holdout_width_h_per_Mpc: float = DEFAULT_HOLDOUT_WIDTH_H_PER_MPC,
    k_ref_h_per_Mpc: float = DEFAULT_K_REF_H_PER_MPC,
) -> dict:
    values = np.asarray(k_max_values_h_per_Mpc, dtype=np.float64)
    values = np.sort(np.unique(np.append(values, float(k_ref_h_per_Mpc))))
    fits = [fit_counterterm(measurement, theory, float(kmax)) for kmax in values]

    ref_index = int(np.argmin(np.abs(values - float(k_ref_h_per_Mpc))))
    ref_fit = fits[ref_index]
    ref_k = ref_fit["k_max_h_per_Mpc"]
    c0 = ref_fit["c_counterterm_Mpch2"]
    sigma0 = ref_fit["sigma_c_counterterm_Mpch2"]

    rows = []
    for fit in fits:
        drift = fit["c_counterterm_Mpch2"] - c0
        holdout = run_holdout_test(
            measurement=measurement,
            theory=theory,
            k_fit_max_h_per_Mpc=fit["k_max_h_per_Mpc"],
            k_holdout_max_h_per_Mpc=k_holdout_max_h_per_Mpc,
            holdout_width_h_per_Mpc=holdout_width_h_per_Mpc,
            fit_result=fit,
        )
        frac_resid = fit["residual_columns"]["fractional_residual"]
        rows.append({
            "k_max_h_per_Mpc": fit["k_max_h_per_Mpc"],
            "k_ref_h_per_Mpc": ref_k,
            "c_counterterm_Mpch2": fit["c_counterterm_Mpch2"],
            "sigma_c_counterterm_Mpch2": fit["sigma_c_counterterm_Mpch2"],
            "chi2": fit["chi2"],
            "chi2_dof": fit["chi2_dof"],
            "chi2_linear": fit["chi2_linear"],
            "chi2_spt": fit["chi2_spt"],
            "chi2_improvement": fit["chi2_improvement"],
            "n_points": fit["n_points"],
            "parameter_drift": drift,
            "parameter_drift_fraction": drift / c0 if c0 != 0.0 else np.nan,
            "parameter_drift_sigma": drift / sigma0 if sigma0 > 0.0 else np.nan,
            "rms_fractional_residual": float(np.sqrt(np.mean(frac_resid**2))),
            "max_abs_fractional_residual": float(np.max(np.abs(frac_resid))),
            "holdout_available": holdout["holdout_available"],
            "n_holdout": holdout["n_holdout"],
            "k_holdout_max_h_per_Mpc": holdout["k_holdout_max_h_per_Mpc"],
            "holdout_width_h_per_Mpc": holdout["holdout_width_h_per_Mpc"],
            "chi2_holdout": holdout["chi2_holdout"],
            "chi2_holdout_dof": holdout["chi2_holdout_dof"],
            "chi2_holdout_spt": holdout["chi2_holdout_spt"],
            "chi2_holdout_improvement": holdout["chi2_holdout_improvement"],
            "rms_fractional_residual_holdout": holdout["rms_fractional_residual_holdout"],
        })

    return {
        "rows": rows,
        "fits": fits,
        "metadata": {
            "observable_type": measurement["metadata"].get("observable_type", "matter_pk_real"),
            "redshift": measurement["metadata"].get("redshift"),
            "weight_mode": fits[0]["weight_mode"],
            "k_ref_h_per_Mpc": float(ref_k),
            "holdout_width_h_per_Mpc": float(holdout_width_h_per_Mpc),
            "reference_c_counterterm_Mpch2": float(c0),
            "reference_sigma_c_counterterm_Mpch2": float(sigma0),
            "units": dict(PROJECT_UNITS),
            "conventions": dict(SHARED_CONVENTIONS),
            "covariance_scope": "Gaussian mode-counting only; no mock-estimated or non-Gaussian covariance is included.",
            "holdout_note": "Each holdout test uses a fixed-width local window just above the fit range, with k_holdout_max = k_fit_max + holdout_width unless an explicit holdout limit is supplied.",
            "validation_summary": {
                "kmax_values_monotonic": bool(np.all(np.diff(values) > 0.0)),
                "all_fitted_coefficients_finite": bool(
                    np.all(np.isfinite([fit["c_counterterm_Mpch2"] for fit in fits]))
                ),
                "all_residual_lengths_match": bool(
                    all(fit["validation_summary"]["residual_length_match"] for fit in fits)
                ),
                "all_fit_uncertainties_finite": bool(
                    np.all(np.isfinite([fit["sigma_c_counterterm_Mpch2"] for fit in fits]))
                ),
            },
        },
    }
