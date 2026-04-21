#!/usr/bin/env python3
"""Stream Abacus HDF5 particle positions and write density_products.npz for make_density_plots.py."""
from pathlib import Path
import h5py
import numpy as np

# --- CONFIG ---
MAP_INPUT      = Path("/n/eisenstein_lab/Everyone/boryanah/AbacusSummit_highbase_c000_ph100/hdf5/z0.500_field_0_10.hdf5")
OUT_DIR        = Path("/n/home03/hbrittain/COSMO/artifacts/density")
DATASET        = "Position"
BOX_SIZE       = 1000.0      # h^-1 Mpc
CHUNK_ROWS     = 5_000_000   # rows per HDF5 read
STRIDE         = 512         # keep 1-in-N particles for the 2D map products
SLAB_AXIS      = None        # None = infer from shortest coordinate extent
SLICE_THICK    = 12.0        # h^-1 Mpc, thin-slab depth for 2D projection
BINS           = 700         # histogram bins per projected axis
DELTA_SCALES   = [500, 250, 125, 50, 25]  # h^-1 Mpc scale sizes
DELTA_SUBCELLS = 8           # each scale-volume is split into this many cells per axis
DELTA_INPUTS   = None        # None = use all *field_*.hdf5 files in MAP_INPUT.parent
# --------------


def stream_sample(path):
    print(f"Opening {path}")
    with h5py.File(path, "r") as f:
        dset = f[DATASET]
        nrows = dset.shape[0]
        mins = np.full(3, np.inf)
        maxs = np.full(3, -np.inf)
        batches = []
        n_batches = (nrows + CHUNK_ROWS - 1) // CHUNK_ROWS
        for i, start in enumerate(range(0, nrows, CHUNK_ROWS), 1):
            stop = min(start + CHUNK_ROWS, nrows)
            batch = dset[start:stop].astype(np.float32) * BOX_SIZE
            mins = np.minimum(mins, batch.min(axis=0))
            maxs = np.maximum(maxs, batch.max(axis=0))
            sampled = batch[(-start) % STRIDE::STRIDE].copy()
            if sampled.size:
                batches.append(sampled)
            if i == 1 or i == n_batches or i % 20 == 0:
                print(f"  batch {i}/{n_batches}  {100*stop/nrows:.1f}%")
    positions = np.concatenate(batches) if batches else np.empty((0, 3), np.float32)
    return positions, mins.astype(np.float64), maxs.astype(np.float64), nrows


def resolve_delta_inputs(map_input: Path) -> list[Path]:
    """Choose which HDF5 files feed the global delta calculation."""
    if DELTA_INPUTS is not None:
        return [Path(path) for path in DELTA_INPUTS]

    inputs = sorted(map_input.parent.glob("*field_*.hdf5"))
    return inputs if inputs else [map_input]


def full_box_bounds() -> tuple[np.ndarray, np.ndarray]:
    """full simulation cube for the delta grid."""
    half = 0.5 * BOX_SIZE
    mins = np.full(3, -half, dtype=np.float64)
    maxs = np.full(3, +half, dtype=np.float64)
    return mins, maxs


def prepare_delta_grids(mins: np.ndarray, maxs: np.ndarray, n_total: int) -> dict:
    """Build the cell layout metadata for each requested delta scale."""
    total_vol = float(np.prod(maxs - mins))
    extents = maxs - mins

    print(f"  box extents: {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} h^-1 Mpc")
    print("  delta histograms (full particle stream / cell):")

    layouts = {}
    for W in DELTA_SCALES:
        n_vols = [int(extents[i] / W) for i in range(3)]
        if any(n == 0 for n in n_vols):
            print(f"    W = {W:5.0f} h^-1 Mpc  ->  skipped (box too small on one axis: {n_vols})")
            continue

        n_cells = [n * DELTA_SUBCELLS for n in n_vols]
        n_cells_tot = int(np.prod(n_cells))
        cell_sizes = extents / np.asarray(n_cells, dtype=np.float64)
        mean_particles = n_total / n_cells_tot
        print(
            f"    W = {W:5.0f} h^-1 Mpc  ->  {n_cells}  "
            f"({mean_particles:.1f} true particles/cell)"
        )

        layouts[W] = {
            "bins": n_cells,
            "range": [(mins[i], maxs[i]) for i in range(3)],
            "n_cells": np.asarray(n_cells, dtype=np.int32),
            "cell_sizes": cell_sizes.astype(np.float32),
            "mean_particles": np.float32(mean_particles),
            "cell_vol": np.float32(total_vol / n_cells_tot),
        }

    return layouts


def stream_delta_histograms(
    paths: list[Path],
    mins: np.ndarray,
    maxs: np.ndarray,
    ) -> tuple[dict, dict, int]:
    """Stream the full particle catalog and accumulate delta counts per cell.

    The requested delta scales are nested exactly, so we only histogram once on
    the finest grid and sum blocks to recover the coarser scales.
    """
    counts_per_file = []
    for path in paths:
        with h5py.File(path, "r") as f:
            nrows = int(f[DATASET].shape[0])
        counts_per_file.append((path, nrows))

    n_total = sum(nrows for _, nrows in counts_per_file)
    layouts = prepare_delta_grids(mins, maxs, n_total)
    if not layouts:
        return {}, {}, n_total

    finest_scale = min(layouts)
    fine_shape = tuple(int(n) for n in layouts[finest_scale]["n_cells"])
    fine_size = int(np.prod(fine_shape))
    fine_counts = np.zeros(fine_size, dtype=np.uint64)

    # These Abacus coordinates already live in box-fraction units [-0.5, 0.5],
    # so we can bin them directly without re-scaling every batch to Mpc first.
    mins_raw = mins / BOX_SIZE
    maxs_raw = maxs / BOX_SIZE
    inv_cell = np.asarray(fine_shape, dtype=np.float64) / (maxs_raw - mins_raw)

    print(f"Streaming full-resolution delta histograms from {len(paths)} file(s)...")
    for file_i, (path, nrows) in enumerate(counts_per_file, 1):
        print(f"  file {file_i}/{len(paths)}: {path.name}  ({nrows:,} particles)")
        with h5py.File(path, "r") as f:
            dset = f[DATASET]
            n_batches = (nrows + CHUNK_ROWS - 1) // CHUNK_ROWS

            for i, start in enumerate(range(0, nrows, CHUNK_ROWS), 1):
                stop = min(start + CHUNK_ROWS, nrows)
                batch = dset[start:stop]

                idx = np.floor((batch - mins_raw) * inv_cell).astype(np.int32)
                for ax, n_cell in enumerate(fine_shape):
                    np.clip(idx[:, ax], 0, n_cell - 1, out=idx[:, ax])

                flat = idx[:, 0] + fine_shape[0] * (idx[:, 1] + fine_shape[1] * idx[:, 2])
                fine_counts += np.bincount(flat, minlength=fine_size).astype(np.uint64)

                if i == 1 or i == n_batches or i % 20 == 0:
                    print(f"    batch {i}/{n_batches}  {100*stop/nrows:.1f}%")

    fine_grid = fine_counts.reshape(fine_shape)
    delta_arrays = {}
    for W, layout in layouts.items():
        coarse_shape = tuple(int(n) for n in layout["n_cells"])
        block_shape = tuple(f // c for f, c in zip(fine_shape, coarse_shape))

        if any(f % c != 0 for f, c in zip(fine_shape, coarse_shape)):
            raise ValueError(f"Delta grids are not nested cleanly for W = {W}")

        counts = fine_grid.reshape(
            coarse_shape[0], block_shape[0],
            coarse_shape[1], block_shape[1],
            coarse_shape[2], block_shape[2],
        ).sum(axis=(1, 3, 5))

        mean_particles = float(layouts[W]["mean_particles"])
        delta = counts.reshape(-1).astype(np.float32) / mean_particles - 1.0
        delta_arrays[W] = delta.astype(np.float32)

    return delta_arrays, layouts, n_total


def make_projection(positions, mins, maxs):
    slab_ax = SLAB_AXIS if SLAB_AXIS is not None else int(np.argmin(maxs - mins))
    proj = [ax for ax in range(3) if ax != slab_ax]
    mid = 0.5 * (mins[slab_ax] + maxs[slab_ax])
    slab_mask = np.abs(positions[:, slab_ax] - mid) < 0.5 * SLICE_THICK
    plane_pts = positions[slab_mask][:, proj]
    if len(plane_pts) == 0:
        print("Warning: empty slice, using full projection")
        plane_pts = positions[:, proj]
    counts, xe, ye = np.histogram2d(
        plane_pts[:, 0], plane_pts[:, 1], bins=BINS,
        range=[[mins[proj[0]], maxs[proj[0]]], [mins[proj[1]], maxs[proj[1]]]]
    )
    return counts, xe, ye, plane_pts, slab_ax


def main():
    out = OUT_DIR / MAP_INPUT.stem
    out.mkdir(parents=True, exist_ok=True)
    positions, mins, maxs, n_total = stream_sample(MAP_INPUT)
    counts, xe, ye, plane_pts, slab_ax = make_projection(positions, mins, maxs)

    delta_inputs = resolve_delta_inputs(MAP_INPUT)
    delta_mins, delta_maxs = full_box_bounds()
    print("Computing delta histograms...")
    delta_hists, delta_layouts, delta_n_total = stream_delta_histograms(
        delta_inputs,
        delta_mins,
        delta_maxs,
    )

    npz = out / "density_products.npz"
    npz_tmp = out / "density_products_tmp.npz"
    np.savez_compressed(npz_tmp,
        raw_counts=counts.astype(np.float32),
        x_edges=xe.astype(np.float32),
        y_edges=ye.astype(np.float32),
        plane_points=plane_pts.astype(np.float32),
        sampled_positions=positions,
        mins=mins,
        maxs=maxs,
        slab_axis=np.int64(slab_ax),
        box_size=np.float64(BOX_SIZE),
        n_particles_total=np.int64(n_total),
        delta_mins=delta_mins,
        delta_maxs=delta_maxs,
        delta_n_particles_total=np.int64(delta_n_total),
        delta_subcells=np.int64(DELTA_SUBCELLS),
        **{f"delta_{W}": arr for W, arr in delta_hists.items()},
        **{f"delta_n_cells_{W}": layout["n_cells"] for W, layout in delta_layouts.items()},
        **{f"delta_cell_sizes_{W}": layout["cell_sizes"] for W, layout in delta_layouts.items()},
        **{f"delta_mean_particles_{W}": layout["mean_particles"] for W, layout in delta_layouts.items()},
    )
    npz_tmp.replace(npz)
    print(f"Wrote {npz}")


if __name__ == "__main__":
    main()
