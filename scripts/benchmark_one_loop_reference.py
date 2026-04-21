#!/usr/bin/env python3
"""Benchmark the repo one-loop SPT implementation against standard formulas."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit.fit_eft import load_measurement_artifact
from theory.header_cosmo import cosmo_from_header
from theory.linear_baseline import make_linear_pk
from theory.spt_one_loop import compute_one_loop


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_p22(
    *,
    k_eval: np.ndarray,
    k_lin: np.ndarray,
    pk_lin: np.ndarray,
    nr: int = 300,
    nx: int = 120,
    r_min: float = 1.0e-4,
    r_max: float = 50.0,
) -> np.ndarray:
    ln_r = np.linspace(np.log(r_min), np.log(r_max), nr)
    r = np.exp(ln_r)
    dr = np.gradient(ln_r) * r
    x, wx = np.polynomial.legendre.leggauss(nx)
    rr = r[:, None]
    xx = x[None, :]

    def plin(k):
        return np.exp(np.interp(np.log(k), np.log(k_lin), np.log(pk_lin)))

    out = np.empty_like(k_eval)
    for i, k in enumerate(k_eval):
        psi = np.sqrt(np.maximum(1.0 + rr**2 - 2.0 * rr * xx, 1.0e-30))
        kernel = (3.0 * rr + 7.0 * xx - 10.0 * rr * xx**2) ** 2 / np.maximum(psi**4, 1.0e-30)
        mu_integrand = plin(k * rr) * plin(k * psi) * kernel
        out[i] = (1.0 / 98.0) * k**3 / (4.0 * np.pi**2) * np.sum(np.sum(mu_integrand * wx, axis=1) * dr)
    return out


def reference_p13(
    *,
    k_eval: np.ndarray,
    k_lin: np.ndarray,
    pk_lin: np.ndarray,
    nr: int = 2000,
    r_min: float = 1.0e-4,
    r_max: float = 50.0,
) -> np.ndarray:
    ln_r = np.linspace(np.log(r_min), np.log(r_max), nr)
    r = np.exp(ln_r)
    dr = np.gradient(ln_r) * r

    def plin(k):
        return np.exp(np.interp(np.log(k), np.log(k_lin), np.log(pk_lin)))

    out = np.empty_like(k_eval)
    for i, k in enumerate(k_eval):
        bracket = (
            12.0 / r**2
            - 158.0
            + 100.0 * r**2
            - 42.0 * r**4
            + (3.0 / r**3) * (r**2 - 1.0) ** 3 * (7.0 * r**2 + 2.0) * np.log(np.abs((1.0 + r) / (1.0 - r)))
        )
        out[i] = (1.0 / 252.0) * k**3 / (4.0 * np.pi**2) * plin(k) * np.sum(plin(k * r) * bracket * dr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z", type=float, default=0.5, help="Redshift to benchmark.")
    parser.add_argument("--nr", type=int, default=128, help="Repo radial integration samples.")
    parser.add_argument("--nx", type=int, default=64, help="Repo angular integration samples.")
    parser.add_argument("--n-bins", type=int, default=80, help="Number of low-k bins to compare.")
    args = parser.parse_args()

    z = float(args.z)
    source_power = ROOT / "data" / f"z{z:.3f}" / "AB" / "power_nfft2048.csv"
    header_path = ROOT / "data" / f"z{z:.3f}" / "AB" / "header"

    measurement = load_measurement_artifact(source_power)
    cosmo = cosmo_from_header(header_path)

    k_lin = np.geomspace(2.0 * np.pi / cosmo["box_size"] * 0.5, 1.0, 600)
    pk_lin = make_linear_pk(k_lin, cosmo)
    k_eval = measurement["k_h_per_Mpc"][: int(args.n_bins)]

    repo = compute_one_loop(k_eval, k_lin, pk_lin, nr=int(args.nr), nx=int(args.nx))
    ref22 = reference_p22(k_eval=k_eval, k_lin=k_lin, pk_lin=pk_lin)
    ref13 = reference_p13(k_eval=k_eval, k_lin=k_lin, pk_lin=pk_lin)

    rows = []
    for i, k in enumerate(k_eval):
        rows.append(
            {
                "k_h_per_Mpc": float(k),
                "repo_P22": float(repo["P22"][i]),
                "ref_P22": float(ref22[i]),
                "repo_P13": float(repo["P13"][i]),
                "ref_P13": float(ref13[i]),
                "P22_fractional_difference": float((repo["P22"][i] - ref22[i]) / ref22[i]),
                "P13_fractional_difference": float((repo["P13"][i] - ref13[i]) / ref13[i]),
            }
        )

    summary = {
        "redshift": z,
        "reference_source": (
            "Standard one-loop formulas matching the explicit expressions in Hikage, Koyama & Heavens "
            "(2017), eqs. (18) and (19), for P22 and P13."
        ),
        "repo_settings": {
            "nr": int(args.nr),
            "nx": int(args.nx),
        },
        "median_abs_fractional_difference_P22": float(np.median(np.abs((repo["P22"] - ref22) / ref22))),
        "max_abs_fractional_difference_P22": float(np.max(np.abs((repo["P22"] - ref22) / ref22))),
        "median_abs_fractional_difference_P13": float(np.median(np.abs((repo["P13"] - ref13) / ref13))),
        "max_abs_fractional_difference_P13": float(np.max(np.abs((repo["P13"] - ref13) / ref13))),
    }

    output_dir = ROOT / "artifacts" / "theory" / f"z{z:.3f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "one_loop_reference_benchmark.csv", rows)
    (output_dir / "one_loop_reference_benchmark.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
