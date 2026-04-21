#!/usr/bin/env python3
"""Render density figures from density_products.npz.

Outputs:
  density_map_raw.png
  density_map_smoothed.png
  zoom_overdense/frame_NNN_WWWW_hmpc.png  (+  .gif if Pillow available)
  delta_distributions.png
  delta_distributions_overlay.png
"""

# ------------------
# Imports
# ------------------
# standard -----
from pathlib import Path
# scientific -----
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# optional -----
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

# ------------------
# Config
# ------------------

NPZ          = Path("/n/home03/hbrittain/COSMO/artifacts/density/z0.500_field_0_10/density_products.npz")
OUT_DIR      = None       # None = same directory as NPZ

CMAP         = "magma"
SMOOTH_SIGMA = 4.0        # h^-1 Mpc, Gaussian smoothing for full-field map

# zoom animation — single overdense target
ZOOM_WIDTHS     = None    # None = log-spaced from ZOOM_MAX down to ZOOM_MIN; or set explicit list
ZOOM_N_FRAMES   = 48
ZOOM_MAX        = 420.0   # h^-1 Mpc, widest frame
ZOOM_MIN        = 5.0     # h^-1 Mpc, narrowest frame
ZOOM_BINS       = 400     # 2D histogram bins per axis inside the zoom box
SEARCH_FRACTION = 0.5     # overdense peak search restricted to central fraction of field
FPS             = 3.0     # GIF frames per second

# global delta distribution — tile full box at each scale W
DELTA_SCALES    = [500, 250, 125, 50, 25]   # h^-1 Mpc scale sizes (must divide box evenly)
DELTA_HIST_BINS = 60

AXES = ("x", "y", "z")

# ------------------
# Helpers
# ------------------

def smooth2d(arr, sigma_px):
    if sigma_px <= 0:
        return arr.astype(np.float32)
    if gaussian_filter is not None:
        return gaussian_filter(arr.astype(np.float32), sigma=sigma_px)
    # fallback: manual separable convolution
    r = max(1, int(3 * sigma_px))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma_px) ** 2); k /= k.sum()
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, "same"), 0, arr.astype(np.float32))
    return np.apply_along_axis(lambda m: np.convolve(m, k, "same"), 1, tmp)


def find_overdense_center(smooth_counts, x_edges, y_edges):
    """Peak of the smoothed projection within the central SEARCH_FRACTION of the field."""
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    xm, ym = 0.5 * (x_edges[0] + x_edges[-1]), 0.5 * (y_edges[0] + y_edges[-1])
    xh = 0.5 * (x_edges[-1] - x_edges[0]) * SEARCH_FRACTION
    yh = 0.5 * (y_edges[-1] - y_edges[0]) * SEARCH_FRACTION
    gx, gy = np.meshgrid(xc, yc, indexing="ij")
    mask = (np.abs(gx - xm) <= xh) & (np.abs(gy - ym) <= yh)
    ix, iy = np.unravel_index(np.nanargmax(np.where(mask, smooth_counts, np.nan)), smooth_counts.shape)
    return float(xc[ix]), float(yc[iy])

# ------------------
# Plots
# ------------------

def plot_full_field(raw_counts, smooth_counts, x_edges, y_edges, out_dir, xlabel, ylabel):
    for arr, fname, title in [
        (raw_counts,    "density_map_raw.png",      "Particle density (raw)"),
        (smooth_counts, "density_map_smoothed.png", f"Particle density (smoothed, sigma = {SMOOTH_SIGMA} h^-1 Mpc)"),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 6), constrained_layout=True)
        im = ax.imshow(np.log10(arr.T + 1), origin="lower", cmap=CMAP, aspect="equal",
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(im, ax=ax, label=r"$\log_{10}(\mathrm{count} + 1)$")
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote full-field maps to {out_dir}")


def zoom_animation(plane_pts, raw_counts, smooth_counts, x_edges, y_edges, out_dir, xlabel, ylabel):
    cx, cy = find_overdense_center(smooth_counts, x_edges, y_edges)
    print(f"  overdense center: ({cx:.1f}, {cy:.1f}) h^-1 Mpc")

    widths   = ZOOM_WIDTHS if ZOOM_WIDTHS is not None else np.geomspace(ZOOM_MAX, ZOOM_MIN, ZOOM_N_FRAMES).tolist()
    zoom_dir = out_dir / "zoom_overdense"
    zoom_dir.mkdir(exist_ok=True)
    frame_paths = []

    # precompute full-field image — reused across every frame
    full_img = np.log10(raw_counts.T + 1)
    full_ext = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    vmax_full = float(full_img.max())

    for i, W in enumerate(widths):
        half = 0.5 * W
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half

        plane_mask = ((plane_pts[:, 0] >= x0) & (plane_pts[:, 0] < x1) &
                      (plane_pts[:, 1] >= y0) & (plane_pts[:, 1] < y1))
        frame_counts, _, _ = np.histogram2d(
            plane_pts[plane_mask, 0], plane_pts[plane_mask, 1],
            bins=ZOOM_BINS, range=[[x0, x1], [y0, y1]]
        )

        fig, (ax_ov, ax_zm) = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

        # left — full field with a red box showing where we're zooming
        im_ov = ax_ov.imshow(full_img, origin="lower", cmap=CMAP, aspect="equal",
                             extent=full_ext, vmin=0, vmax=vmax_full)
        ax_ov.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   linewidth=2, edgecolor="red", facecolor="none"))
        ax_ov.set(title="Full field", xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(im_ov, ax=ax_ov, label=r"$\log_{10}(\mathrm{count} + 1)$")

        # right — zoomed density
        im_zm = ax_zm.imshow(np.log10(frame_counts.T + 1), origin="lower", cmap=CMAP,
                             aspect="equal", extent=[x0, x1, y0, y1])
        ax_zm.set(title=f"Zoom: {W:.0f} h^-1 Mpc wide", xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(im_zm, ax=ax_zm, label=r"$\log_{10}(\mathrm{count} + 1)$")

        p = zoom_dir / f"frame_{i:03d}_{int(round(W)):04d}_hmpc.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(p)

    if Image is not None and frame_paths:
        frames = [Image.open(p) for p in frame_paths]
        gif = zoom_dir / "density_zoom_overdense.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=max(1, int(1000 / FPS)), loop=0)
        for f in frames:
            f.close()
        print(f"  wrote {len(frame_paths)} frames + GIF to {zoom_dir}")


def plot_delta_distributions(
    delta_arrays: dict,
    mins: np.ndarray,
    maxs: np.ndarray,
    out_dir: Path,
    delta_subcells: int,
    delta_means: dict,
) -> None:
    """Plot pre-computed delta distributions from density_products.npz.

    Args:
        delta_arrays: dict mapping W (h^-1 Mpc) -> 1-D float32 array of delta values
        mins: simulation box lower bounds
        maxs: simulation box upper bounds
        out_dir: output directory for the figure
    """
    # only plot scales that were actually computed (some may have been skipped)
    present_scales = [W for W in DELTA_SCALES if W in delta_arrays]
    all_deltas     = [delta_arrays[W] for W in present_scales]

    if not present_scales:
        print("  no delta arrays found in NPZ")
        return

    # shared x range across all panels, clipped away from the wild far tail
    xmin = -1.05
    xmax = max(float(np.percentile(delta, 99.5)) for delta in all_deltas)
    colors = plt.cm.cividis(np.linspace(0.15, 0.9, len(present_scales)))

    n_scales = len(present_scales)
    fig, axes = plt.subplots(1, n_scales, figsize=(3.5 * n_scales, 4.5),
                             constrained_layout=True, sharey=False)
    if n_scales == 1:
        axes = [axes]

    for ax, W, delta, color in zip(axes, present_scales, all_deltas, colors):
        ax.hist(delta, bins=DELTA_HIST_BINS, range=(xmin, xmax),
                density=True, color=color, alpha=0.95)
        ax.axvline(0, color="#c1121f", lw=1.5, ls="--")
        ax.set_xlim(xmin, xmax)

        n_vols   = [int((maxs[i] - mins[i]) / W) for i in range(3)]
        n_cells  = [n * delta_subcells for n in n_vols]
        cell_mpc = (maxs - mins) / np.asarray(n_cells, dtype=np.float64)
        mean_particles = float(delta_means.get(W, np.nan))
        cell_str = " x ".join(f"{size:.2f}" for size in cell_mpc)
        ax.set(title=f"{W:.0f} h^-1 Mpc scale\n({cell_str} Mpc cells, {mean_particles:.0f} p/cell)",
               xlabel=r"$\delta$  (density contrast)",
               ylabel="probability density" if ax is axes[0] else "")

    fig.suptitle(r"Global density contrast  $\delta = \rho\,/\,\bar{\rho}_{\rm global} - 1$",
                 fontsize=12)
    p = out_dir / "delta_distributions.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")

    # direct shape comparison on one set of axes
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    edges = np.linspace(xmin, xmax, DELTA_HIST_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for W, delta, color in zip(present_scales, all_deltas, colors):
        hist, _ = np.histogram(delta, bins=edges, density=True)
        mean_particles = float(delta_means.get(W, np.nan))
        ax.plot(
            centers,
            hist,
            color=color,
            lw=2.0,
            label=f"{W:.0f} h^-1 Mpc  ({mean_particles:.0f} p/cell)",
        )

    ax.axvline(0, color="#c1121f", lw=1.5, ls="--")
    ax.set(
        xlim=(xmin, xmax),
        xlabel=r"$\delta$  (density contrast)",
        ylabel="probability density",
        title="Density contrast distribution shape vs. scale",
    )
    ax.legend(frameon=False)
    p = out_dir / "delta_distributions_overlay.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")

# ------------------
# Main
# ------------------

def main():
    data      = np.load(NPZ)
    raw       = data["raw_counts"]
    x_edges   = data["x_edges"]
    y_edges   = data["y_edges"]
    plane_pts = data["plane_points"]
    mins      = data["mins"]
    maxs      = data["maxs"]
    slab_axis = int(data["slab_axis"])
    n_total   = int(data["n_particles_total"])
    delta_arrays = {W: data[f"delta_{W}"] for W in DELTA_SCALES if f"delta_{W}" in data}
    delta_mins = data["delta_mins"] if "delta_mins" in data else mins
    delta_maxs = data["delta_maxs"] if "delta_maxs" in data else maxs
    delta_n_total = int(data["delta_n_particles_total"]) if "delta_n_particles_total" in data else n_total
    data_subcells = int(data["delta_subcells"]) if "delta_subcells" in data else 8
    delta_means = {
        W: float(data[f"delta_mean_particles_{W}"])
        for W in DELTA_SCALES
        if f"delta_mean_particles_{W}" in data
    }

    out_dir = Path(NPZ).parent if OUT_DIR is None else Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    proj   = [ax for ax in range(3) if ax != slab_axis]
    xlabel = f"comoving {AXES[proj[0]]}  [h^-1 Mpc]"
    ylabel = f"comoving {AXES[proj[1]]}  [h^-1 Mpc]"

    total_vol = float(np.prod(delta_maxs - delta_mins))
    n_bar     = delta_n_total / total_vol
    print(f"Delta volume: {total_vol:.3e} (h^-1 Mpc)^3   |   n_bar = {n_bar:.3e} (h^-1 Mpc)^-3")

    sigma_px     = SMOOTH_SIGMA * (len(x_edges) - 1) / float(x_edges[-1] - x_edges[0])
    smooth       = smooth2d(raw, sigma_px)

    print("Plotting full-field maps...")
    plot_full_field(raw, smooth, x_edges, y_edges, out_dir, xlabel, ylabel)

    print("Rendering zoom animation...")
    zoom_animation(plane_pts, raw, smooth, x_edges, y_edges, out_dir, xlabel, ylabel)

    print("Plotting delta distributions...")
    plot_delta_distributions(
        delta_arrays,
        delta_mins,
        delta_maxs,
        out_dir,
        data_subcells,
        delta_means,
    )


if __name__ == "__main__":
    main()
