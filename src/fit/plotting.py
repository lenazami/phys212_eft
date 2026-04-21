from __future__ import annotations

# ------------------
# Imports
# ------------------

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ------------------
# Constants
# ------------------

# plot colors!
COLORS = {
    "measured": "#1e293b",
    "linear":   "#3b82f6",
    "spt":      "#8b5cf6",
    "eft":      "#ec4899",
    "loop":     "#14b8a6",
    "p22":      "#6366f1",
    "p13":      "#a78bfa",
    "ctr":      "#f472b6",
    "fit":      "#6366f1",
    "holdout":  "#f472b6",
}

SCALE_MARKER_COLORS = {
    "k_ref":     "#94a3b8",
    "k_fit":     "#3b82f6",
    "k_stab":    "#ec4899",
    "k_nyquist": "#8b5cf6",
    "k_nl":      "#a855f7",
}

SCALE_MARKER_LABELS = {
    "k_ref":     r"$k_{\rm ref}$",
    "k_fit":     r"$k_{\rm fit}$",
    "k_stab":    r"$k_{\rm stab}$",
    "k_nyquist": r"$k_{\rm Nyq}$",
    "k_nl":      r"$k_{\rm NL}$",
}

# ------------------
# Functions
# ------------------
def _save(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _delta_squared(k: np.ndarray, pk: np.ndarray) -> np.ndarray:
    return k**3 * pk / (2.0 * np.pi**2)


def _add_scale_markers(ax: plt.Axes, scale_markers: dict[str, float] | None) -> None:
    if not scale_markers:
        return
    for key, value in scale_markers.items():
        if not np.isfinite(value):
            continue
        ax.axvline(
            float(value),
            color=SCALE_MARKER_COLORS.get(key, "#94a3b8"),
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            label=SCALE_MARKER_LABELS.get(key, key),
        )


def _scan_kmax(scan_rows) -> np.ndarray:
    return np.asarray([
        row["k_max"] if "k_max" in row else row["k_max_h_per_Mpc"]
        for row in scan_rows
    ])


def save_fractional_residuals(
    *,
    k,
    p_meas,
    p11,
    p_1loop,
    p_eft,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0, color=COLORS["measured"], linewidth=0.8)
    ax.semilogx(k, (p_meas - p11) / p_meas, color=COLORS["linear"], label="Linear")
    ax.semilogx(k, (p_meas - p_1loop) / p_meas, color=COLORS["spt"], label="One-loop SPT")
    ax.semilogx(k, (p_meas - p_eft) / p_meas, color=COLORS["eft"], label="EFT")
    ax.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
    ax.set_ylabel("fractional residual")
    ax.grid(alpha=0.2)
    ax.legend()
    return _save(fig, output_path)


def save_main_spectrum_comparison(
    *,
    k,
    p_meas,
    p11,
    p_1loop,
    p_eft,
    output_path: Path,
    scale_markers: dict[str, float] | None = None,
    residual_yscale: str = "symlog",
    residual_ylim: tuple[float, float] | None = None,
) -> Path:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    ax1.loglog(k, p_meas, color=COLORS["measured"], linewidth=1.8, label="Measured")
    ax1.loglog(k, p11, color=COLORS["linear"], linewidth=1.6, label="Linear")
    ax1.loglog(k, p_1loop, color=COLORS["spt"], linewidth=1.6, label="One-loop SPT")
    ax1.loglog(k, p_eft, color=COLORS["eft"], linewidth=1.8, label="One-loop EFT")
    _add_scale_markers(ax1, scale_markers)
    ax1.set_ylabel(r"$P(k)\ [({\rm Mpc}/h)^3]$")
    ax1.grid(alpha=0.2)
    ax1.legend(loc="lower left", ncol=2, fontsize=9)

    ax2.axhline(0.0, color=COLORS["measured"], linewidth=0.8)
    ax2.semilogx(k, (p11 - p_meas) / p_meas, color=COLORS["linear"], linewidth=1.6, label="Linear")
    ax2.semilogx(k, (p_1loop - p_meas) / p_meas, color=COLORS["spt"], linewidth=1.6, label="One-loop SPT")
    ax2.semilogx(k, (p_eft - p_meas) / p_meas, color=COLORS["eft"], linewidth=1.8, label="One-loop EFT")
    _add_scale_markers(ax2, scale_markers)
    ax2.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
    ax2.set_ylabel(r"$(P_{\rm model} - P_{\rm sim})/P_{\rm sim}$")
    if residual_yscale == "symlog":
        ax2.set_yscale("symlog", linthresh=1.0e-2)
    else:
        ax2.set_yscale(residual_yscale)
    if residual_ylim is not None:
        ax2.set_ylim(*residual_ylim)
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper right", fontsize=9)
    return _save(fig, output_path)


def save_loop_decomposition(
    *,
    k,
    p22,
    p13,
    p_loop,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0, color=COLORS["measured"], linewidth=0.8)
    ax.semilogx(k, p22, color=COLORS["p22"], label="P22")
    ax.semilogx(k, p13, color=COLORS["p13"], label="P13")
    ax.semilogx(k, p_loop, color=COLORS["loop"], linewidth=1.8, label=r"$P_{1\mathrm{loop}}-P_{11}$")
    ax.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
    ax.set_ylabel(r"$P(k)\ [({\rm Mpc}/h)^3]$")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(alpha=0.2)
    ax.legend()
    return _save(fig, output_path)


def save_counterterm_contribution(
    *,
    k,
    p_ctr,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0, color=COLORS["measured"], linewidth=0.8)
    ax.semilogx(k, p_ctr, color=COLORS["ctr"], label=r"$P_{\rm ctr}$")
    ax.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
    ax.set_ylabel(r"$P_{\rm ctr}(k)\ [({\rm Mpc}/h)^3]$")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(alpha=0.2)
    ax.legend()
    return _save(fig, output_path)


def save_stability_scan(
    *,
    scan_rows,
    output_path: Path,
    selected_k_stab: float | None = None,
) -> Path:
    kmax = _scan_kmax(scan_rows)
    cvals = np.asarray([row["c_counterterm_Mpch2"] for row in scan_rows])
    sigma_c = np.asarray([row["sigma_c_counterterm_Mpch2"] for row in scan_rows])
    drift_sigma = np.asarray([row["parameter_drift_sigma"] for row in scan_rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    ax1.errorbar(
        kmax,
        cvals,
        yerr=sigma_c,
        color=COLORS["ctr"],
        marker="o",
        linewidth=1.8,
        capsize=3,
    )
    if selected_k_stab is not None:
        ax1.axvline(selected_k_stab, color=COLORS["eft"], linestyle="--", linewidth=1.2)
    ax1.set_ylabel(r"$c_{\rm ctr}\ [({\rm Mpc}/h)^2]$")
    ax1.grid(alpha=0.2)

    ax2.axhline(0.0, color=COLORS["measured"], linewidth=0.8)
    ax2.axhline(1.0, color="#cbd5e1", linestyle="--", linewidth=0.9)
    ax2.axhline(-1.0, color="#cbd5e1", linestyle="--", linewidth=0.9)
    ax2.plot(kmax, drift_sigma, color=COLORS["linear"], marker="o", linewidth=1.8)
    if selected_k_stab is not None:
        ax2.axvline(selected_k_stab, color=COLORS["eft"], linestyle="--", linewidth=1.2)
    ax2.set_xlabel(r"$k_{\max}\ [h/{\rm Mpc}]$")
    ax2.set_ylabel(r"$\Delta c / \sigma(c_{\rm ref})$")
    ax2.grid(alpha=0.2)
    return _save(fig, output_path)


def save_cumulative_chi2(
    *,
    scan_rows,
    output_path: Path,
    selected_k_stab: float | None = None,
) -> Path:
    kmax = _scan_kmax(scan_rows)
    chi2_eft = np.asarray([row["chi2_dof"] for row in scan_rows])
    chi2_linear = np.asarray([
        row["chi2_linear"] / max(row["n_points"] - 1, 1) if np.isfinite(row["chi2_linear"]) else np.nan
        for row in scan_rows
    ])
    chi2_spt = np.asarray([
        row["chi2_spt"] / max(row["n_points"] - 1, 1) if np.isfinite(row["chi2_spt"]) else np.nan
        for row in scan_rows
    ])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(kmax, chi2_linear, color=COLORS["linear"], marker="d", label="Linear")
    ax.plot(kmax, chi2_spt, color=COLORS["spt"], marker="s", label="One-loop SPT")
    ax.plot(kmax, chi2_eft, color=COLORS["eft"], marker="o", label="One-loop EFT")
    ax.axhline(1.0, color="#cbd5e1", linestyle="--", linewidth=0.9)
    if selected_k_stab is not None:
        ax.axvline(selected_k_stab, color=COLORS["eft"], linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$k_{\max}\ [h/{\rm Mpc}]$")
    ax.set_ylabel(r"$\chi^2/{\rm dof}$")
    ax.grid(alpha=0.2)
    ax.legend()
    return _save(fig, output_path)


def save_holdout_scan(
    *,
    scan_rows,
    output_path: Path,
    selected_k_stab: float | None = None,
) -> Path:
    kmax = _scan_kmax(scan_rows)
    holdout_chi2 = np.asarray([row["chi2_holdout_dof"] for row in scan_rows])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(kmax, holdout_chi2, color=COLORS["holdout"], marker="^", s=90, label="EFT local holdout")
    ax.axhline(1.0, color="#cbd5e1", linestyle="--", linewidth=0.9)
    if selected_k_stab is not None:
        ax.axvline(selected_k_stab, color=COLORS["eft"], linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$k_{\max}\ [h/{\rm Mpc}]$")
    ax.set_ylabel(r"local holdout $\chi^2/{\rm dof}$")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    return _save(fig, output_path)


def save_dimensionless_power(
    *,
    k,
    p_meas,
    p11,
    p_1loop,
    p_eft,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k, _delta_squared(k, p_meas), color=COLORS["measured"], label="Measured")
    ax.loglog(k, _delta_squared(k, p11), color=COLORS["linear"], label="Linear")
    ax.loglog(k, _delta_squared(k, p_1loop), color=COLORS["spt"], label="One-loop SPT")
    ax.loglog(k, _delta_squared(k, p_eft), color=COLORS["eft"], label="EFT")
    ax.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
    ax.set_ylabel(r"$\Delta^2(k)$")
    ax.grid(alpha=0.2)
    ax.legend()
    return _save(fig, output_path)


def save_theory_validation(
    *,
    validation: dict,
    output_path: Path,
) -> Path:
    cols = validation["columns"]
    k = cols["k_h_per_Mpc"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    ax1.plot(k, cols["P13_over_k2P11"], color=COLORS["spt"], marker="o", label=r"measured $P_{13}/(k^2 P_{11})$")
    ax1.plot(k, cols["expected_P13_over_k2P11"], color=COLORS["linear"], linestyle="--", linewidth=1.2, label="soft-limit expectation")
    ax1.set_ylabel(r"$P_{13}/(k^2 P_{11})$")
    ax1.grid(alpha=0.2)
    ax1.legend()

    ax2.loglog(k, cols["P22"], color=COLORS["p22"], marker="o")
    ax2.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
    ax2.set_ylabel(r"$P_{22}(k)$")
    ax2.grid(alpha=0.2)
    return _save(fig, output_path)


def save_redshift_stability_summary(
    *,
    summary_rows,
    output_path: Path,
) -> Path:
    z = np.asarray([row["redshift"] for row in summary_rows])
    k_stab = np.asarray([row["selected_k_stab"] for row in summary_rows])
    cvals = np.asarray([row["c_counterterm_Mpch2"] for row in summary_rows])
    sigma_c = np.asarray([row["sigma_c_counterterm_Mpch2"] for row in summary_rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    ax1.plot(z, k_stab, color=COLORS["fit"], marker="o")
    ax1.set_ylabel(r"$k_{\rm stab}\ [h/{\rm Mpc}]$")
    ax1.grid(alpha=0.2)

    ax2.errorbar(z, cvals, yerr=sigma_c, color=COLORS["eft"], marker="o", capsize=3)
    ax2.set_xlabel("redshift")
    ax2.set_ylabel(r"$c_{\rm ctr}\ [({\rm Mpc}/h)^2]$")
    ax2.grid(alpha=0.2)
    return _save(fig, output_path)


def save_all_power_figures(
    *,
    output_dir: Path,
    fit: dict,
    scan: dict,
    scale_markers: dict[str, float] | None = None,
    selected_k_stab: float | None = None,
    theory_validation: dict | None = None,
    include_extended: bool = False,
) -> dict[str, str]:
    cols = fit["fit_columns"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "paper_comparison": save_main_spectrum_comparison(
            k=cols["k_h_per_Mpc"],
            p_meas=cols["P_meas"],
            p11=cols["P11"],
            p_1loop=cols["P_1loop"],
            p_eft=cols["P_eft"],
            scale_markers=scale_markers,
            output_path=output_dir / "paper_comparison.png",
        ),
        "loop_decomposition": save_loop_decomposition(
            k=cols["k_h_per_Mpc"],
            p22=cols["P22"],
            p13=cols["P13"],
            p_loop=cols["P_1loop"] - cols["P11"],
            output_path=output_dir / "loop_decomposition.png",
        ),
        "stability_scan": save_stability_scan(
            scan_rows=scan["rows"],
            selected_k_stab=selected_k_stab,
            output_path=output_dir / "stability_scan.png",
        ),
        "cumulative_chi2": save_cumulative_chi2(
            scan_rows=scan["rows"],
            selected_k_stab=selected_k_stab,
            output_path=output_dir / "cumulative_chi2.png",
        ),
        "holdout_scan": save_holdout_scan(
            scan_rows=scan["rows"],
            selected_k_stab=selected_k_stab,
            output_path=output_dir / "holdout_scan.png",
        ),
    }

    if include_extended:
        paths["counterterm"] = save_counterterm_contribution(
            k=cols["k_h_per_Mpc"],
            p_ctr=cols["P_ctr"],
            output_path=output_dir / "counterterm.png",
        )
        paths["dimensionless"] = save_dimensionless_power(
            k=cols["k_h_per_Mpc"],
            p_meas=cols["P_meas"],
            p11=cols["P11"],
            p_1loop=cols["P_1loop"],
            p_eft=cols["P_eft"],
            output_path=output_dir / "dimensionless.png",
        )
        if theory_validation is not None:
            paths["theory_validation"] = save_theory_validation(
                validation=theory_validation,
                output_path=output_dir / "theory_validation.png",
            )
    return {name: str(path) for name, path in paths.items()}
