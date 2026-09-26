#!/usr/bin/env python3
"""Improved 2-D UMAP views from the ALREADY-SAVED full-corpus coordinates and
HDBSCAN labels. This is a pure re-visualisation step.

It does NOT rerun UMAP, HDBSCAN, or embeddings and does NOT modify any existing
analytical result or figure. It reads two saved, row-aligned artifacts:
  results/umap_2d_visualization_coordinates.npy   (float32, 902144 x 2)
  results/hdbscan_labels.parquet                  (label; -1 = noise)
and writes new figures + one JSON sidecar into figures/improved_umap_views/.

Why: a very small number of extreme UMAP coordinates (one document sits at
UMAP-1 ~= -357 while the central 99.9% of UMAP-1 lies within ~[-17, 17]) stretch
the axes so far that almost the whole 902,144-document corpus is compressed into a
narrow strip, and the assigned-vs-noise scatter overplots ~10^6 points. The fix:
report the coordinate distribution, then render density (hexbin, log colour scale)
inside robust percentile viewports, keeping one full-extent map and one dedicated
extreme-coordinate view so nothing is deleted or hidden.

Figures (title + axis/tick labels + colourbar label only; no interpretation, no
cluster naming):
  fig1_full_extent_density.png            all documents, full extent
  fig2_central_99p9_density.png           UMAP-1/2 each 0.05th-99.95th pct
  fig3_central_99p0_density.png           UMAP-1/2 each 0.5th-99.5th pct
  fig4_assigned_vs_noise_central_99p9.png assigned (>=0) vs noise (-1), shared scale
  fig5_extreme_outside_99p9.png           documents outside the 99.9% viewport
  umap_view_summary.json                  quantiles, viewport bounds, in/out counts

No GPU required. Run with the same WSL2 rapids venv that produced the baseline so
the provenance manifests (software block, repo-relative paths) match the project norm.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# Reuse the baseline run's provenance manifest writer / hashing / git helpers so the new
# sidecars are byte-for-byte consistent with the rest of the deliverable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_full_corpus_umap_hdbscan as R  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "figures", "improved_umap_views")

COORDS_NPY = os.path.join(RES, "umap_2d_visualization_coordinates.npy")
LABELS_PARQUET = os.path.join(RES, "hdbscan_labels.parquet")

CMAP = "viridis"
XLAB = "UMAP-1 (2-D visualisation)"
YLAB = "UMAP-2 (2-D visualisation)"
CBAR = "Documents per hexbin (log scale)"

# The exact quantiles the task requests, in order.
QLABELS = [
    "min",
    "0.01%",
    "0.05%",
    "0.1%",
    "0.5%",
    "1%",
    "median",
    "99%",
    "99.5%",
    "99.9%",
    "99.95%",
    "99.99%",
    "max",
]
QVALS = [0, 0.01, 0.05, 0.1, 0.5, 1, 50, 99, 99.5, 99.9, 99.95, 99.99, 100]


def qdict(v: np.ndarray) -> dict:
    """Ordered percentile dict for one coordinate axis (float64 for precision)."""
    ps = np.percentile(v.astype(np.float64), QVALS)
    return {lab: float(p) for lab, p in zip(QLABELS, ps)}


def figsize_for(dx: float, dy: float, target=9.0, cbar=1.9, lo=5.5, hi=12.0):
    """Figure size whose plot area follows the viewport's data ranges (the larger
    data span maps to `target` inches), so proportions reflect the coordinates and
    a handful of extremes cannot dictate the shape. `cbar` leaves room for the bar."""
    if dx <= 0 or dy <= 0:
        return (target + cbar, target)
    if dx >= dy:
        w, h = target, target * dy / dx
    else:
        w, h = target * dx / dy, target
    w = min(max(w, lo), hi)
    h = min(max(h, lo), hi)
    return (w + cbar, h)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq
    from matplotlib.colors import LogNorm

    os.makedirs(OUT, exist_ok=True)

    # ---- load saved artifacts (read-only) + verify row alignment ----
    coords = np.load(COORDS_NPY)
    assert coords.ndim == 2 and coords.shape[1] == 2, f"coords shape {coords.shape}"
    lt = pq.read_table(LABELS_PARQUET)
    row = lt.column("row").to_numpy()
    labels = lt.column("label").to_numpy()
    N = coords.shape[0]
    assert labels.shape[0] == N, "labels/coords row-count mismatch"
    assert np.array_equal(row, np.arange(N)), "labels 'row' is not 0..N-1 (alignment)"
    assert np.isfinite(coords).all(), "non-finite coordinate present"

    x = coords[:, 0].astype(np.float64)
    y = coords[:, 1].astype(np.float64)
    assigned_mask = labels >= 0
    noise_mask = labels == -1
    n_assigned = int(assigned_mask.sum())
    n_noise = int(noise_mask.sum())
    print(f"N={N}  assigned(>=0)={n_assigned}  noise(-1)={n_noise}")

    # ---- quantiles ----
    q1 = qdict(x)
    q2 = qdict(y)

    def show_q(name, q):
        print(f"\n{name} quantiles:")
        for lab in QLABELS:
            print(f"  {lab:>7}: {q[lab]:+.5f}")

    show_q("UMAP-1", q1)
    show_q("UMAP-2", q2)

    # ---- viewports (robust percentile boxes; a doc is inside iff BOTH axes are) ----
    def box(lo_lab, hi_lab):
        return (
            (q1[lo_lab], q1[hi_lab]),  # UMAP-1 (lo, hi)
            (q2[lo_lab], q2[hi_lab]),  # UMAP-2 (lo, hi)
        )

    def inside_mask(b):
        (lo1, hi1), (lo2, hi2) = b
        return (x >= lo1) & (x <= hi1) & (y >= lo2) & (y <= hi2)

    box_999 = box("0.05%", "99.95%")  # central 99.9%
    box_99 = box("0.5%", "99.5%")  # central 99%
    in999 = inside_mask(box_999)
    in99 = inside_mask(box_99)
    n_in999, n_in99 = int(in999.sum()), int(in99.sum())
    n_out999, n_out99 = N - n_in999, N - n_in99

    def pct(n):
        return round(100.0 * n / N, 4)

    print(f"\nCentral 99.9% box UMAP-1={box_999[0]} UMAP-2={box_999[1]}")
    print(
        f"  inside={n_in999:,} ({pct(n_in999)}%)  outside={n_out999:,} ({pct(n_out999)}%)"
    )
    print(f"Central 99% box   UMAP-1={box_99[0]} UMAP-2={box_99[1]}")
    print(
        f"  inside={n_in99:,} ({pct(n_in99)}%)  outside={n_out99:,} ({pct(n_out99)}%)"
    )

    # panel counts for fig 4 (within the 99.9% viewport)
    a999 = in999 & assigned_mask
    z999 = in999 & noise_mask
    na, nz = int(a999.sum()), int(z999.sum())

    # extreme (outside 99.9%) census
    out999 = ~in999
    xo, yo = x[out999], y[out999]
    extreme = {
        "count": int(out999.sum()),
        "pct_of_corpus": pct(int(out999.sum())),
        "umap_1_range": [float(xo.min()), float(xo.max())] if xo.size else None,
        "umap_2_range": [float(yo.min()), float(yo.max())] if yo.size else None,
    }

    written = []  # (abspath, notes) for manifests

    def finish(fig, path, note):
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append((path, note))
        print(f"wrote {os.path.basename(path)}")

    # ---- FIG 1: full-extent density (all documents; extremes remain visible) ----
    fig, ax = plt.subplots(figsize=(14, 6))
    hb = ax.hexbin(
        x, y, gridsize=500, cmap=CMAP, mincnt=1, linewidths=0, norm=LogNorm(vmin=1)
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(CBAR)
    ax.set_xlabel(XLAB)
    ax.set_ylabel(YLAB)
    ax.set_title(f"Full 2-D UMAP density - all documents (N={N:,})")
    fig.tight_layout()
    finish(
        fig,
        os.path.join(OUT, "fig1_full_extent_density.png"),
        "Full-extent 2-D UMAP hexbin density over all documents (log colour "
        "scale); extreme coordinates deliberately left visible.",
    )

    # ---- FIG 2 & 3: central density maps (equal aspect, viewport-sized) ----
    def central_map(b, mask, n_in, n_out, fname, tag):
        (lo1, hi1), (lo2, hi2) = b
        fig, ax = plt.subplots(figsize=figsize_for(hi1 - lo1, hi2 - lo2))
        hb = ax.hexbin(
            x[mask],
            y[mask],
            gridsize=480,
            extent=(lo1, hi1, lo2, hi2),
            cmap=CMAP,
            mincnt=1,
            linewidths=0,
            norm=LogNorm(vmin=1),
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label(CBAR)
        ax.set_xlim(lo1, hi1)
        ax.set_ylim(lo2, hi2)
        ax.set_aspect("equal")
        ax.set_xlabel(XLAB)
        ax.set_ylabel(YLAB)
        ax.set_title(
            f"Central {tag} viewport - {n_in:,} docs shown "
            f"({pct(n_in)}%); {n_out:,} outside"
        )
        fig.tight_layout()
        finish(
            fig,
            os.path.join(OUT, fname),
            f"Central {tag} viewport 2-D UMAP hexbin density (log colour "
            f"scale); axis limits = per-axis "
            f"{'0.05th-99.95th' if tag == '99.9%' else '0.5th-99.5th'} "
            f"percentiles; {n_in:,} docs inside, {n_out:,} outside (not removed).",
        )

    central_map(
        box_999, in999, n_in999, n_out999, "fig2_central_99p9_density.png", "99.9%"
    )
    central_map(box_99, in99, n_in99, n_out99, "fig3_central_99p0_density.png", "99%")

    # ---- FIG 4: assigned vs noise, central 99.9%, directly comparable ----
    (lo1, hi1), (lo2, hi2) = box_999
    ext = (lo1, hi1, lo2, hi2)
    G = 420
    wsize = figsize_for(hi1 - lo1, hi2 - lo2, target=6.5, cbar=0.0)
    fig, (axa, axn) = plt.subplots(
        1, 2, figsize=(2 * wsize[0] + 2.4, wsize[1] + 0.6), sharex=True, sharey=True
    )
    hba = axa.hexbin(
        x[a999], y[a999], gridsize=G, extent=ext, cmap=CMAP, mincnt=1, linewidths=0
    )
    hbn = axn.hexbin(
        x[z999], y[z999], gridsize=G, extent=ext, cmap=CMAP, mincnt=1, linewidths=0
    )
    vmax = max(
        float(hba.get_array().max()) if hba.get_array().size else 1.0,
        float(hbn.get_array().max()) if hbn.get_array().size else 1.0,
    )
    shared = LogNorm(vmin=1, vmax=vmax)  # identical colour scale -> comparable
    hba.set_norm(shared)
    hbn.set_norm(shared)
    for ax in (axa, axn):
        ax.set_xlim(lo1, hi1)
        ax.set_ylim(lo2, hi2)
        ax.set_aspect("equal")
        ax.set_xlabel(XLAB)
    axa.set_ylabel(YLAB)
    axa.set_title(f"Assigned (label >= 0): {na:,}")
    axn.set_title(f"Noise (label = -1): {nz:,}")
    cb = fig.colorbar(hbn, ax=[axa, axn], fraction=0.046, pad=0.04)
    cb.set_label(CBAR)
    fig.suptitle(
        "Central 99.9% viewport - HDBSCAN assigned vs noise (shared density scale)"
    )
    finish(
        fig,
        os.path.join(OUT, "fig4_assigned_vs_noise_central_99p9.png"),
        f"Assigned (>=0, n={na:,}) vs noise (-1, n={nz:,}) within the central "
        f"99.9% viewport; identical x/y limits, hexbin grid, and shared log "
        f"colour scale (vmax={int(vmax)}) so the two panels are directly "
        f"comparable.",
    )

    # ---- FIG 5: extreme-coordinate view (outside the 99.9% viewport) ----
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(xo, yo, s=14, c="#1f77b4", alpha=0.6, linewidths=0, marker=".")
    ax.set_xlabel(XLAB)
    ax.set_ylabel(YLAB)
    ax.set_title(f"Documents outside central 99.9% viewport (n={extreme['count']:,})")
    fig.tight_layout()
    finish(
        fig,
        os.path.join(OUT, "fig5_extreme_outside_99p9.png"),
        f"Documents outside the central 99.9% viewport (n={extreme['count']:,}); "
        f"shown as individual points to make the extreme coordinates visible.",
    )

    # ---- JSON sidecar ----
    summary = {
        "task": "improved 2-D UMAP views from saved coordinates + HDBSCAN labels "
        "(no recomputation of UMAP / HDBSCAN / embeddings)",
        "timestamp": R.utcnow(),
        "git_commit": R.git_short_head(),
        "source": {
            "coords_npy": R.relpath(COORDS_NPY),
            "coords_sha256": R.sha256_file(COORDS_NPY),
            "labels_parquet": R.relpath(LABELS_PARQUET),
            "labels_sha256": R.sha256_file(LABELS_PARQUET),
            "n_documents": N,
        },
        "label_summary": {"assigned_ge0": n_assigned, "noise_neg1": n_noise},
        "quantiles": {
            "percentiles_reported": QLABELS,
            "umap_1": q1,
            "umap_2": q2,
        },
        "viewports": {
            "central_99_9": {
                "definition": "UMAP-1 and UMAP-2 each 0.05th-99.95th percentile",
                "umap_1_range": [box_999[0][0], box_999[0][1]],
                "umap_2_range": [box_999[1][0], box_999[1][1]],
                "documents_inside": n_in999,
                "pct_inside": pct(n_in999),
                "documents_outside": n_out999,
                "pct_outside": pct(n_out999),
                "assigned_inside": na,
                "noise_inside": nz,
            },
            "central_99": {
                "definition": "UMAP-1 and UMAP-2 each 0.5th-99.5th percentile",
                "umap_1_range": [box_99[0][0], box_99[0][1]],
                "umap_2_range": [box_99[1][0], box_99[1][1]],
                "documents_inside": n_in99,
                "pct_inside": pct(n_in99),
                "documents_outside": n_out99,
                "pct_outside": pct(n_out99),
            },
        },
        "extreme_outside_central_99_9": extreme,
        "figures": [R.relpath(p) for p, _ in written],
        "notes": "Counts inside a 2-D percentile box are less than the nominal "
        "per-axis coverage because a document must satisfy BOTH axis "
        "ranges. No documents were removed or altered; outside documents "
        "are shown in fig1 (full extent) and fig5 (extreme view).",
    }
    json_path = os.path.join(OUT, "umap_view_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {os.path.basename(json_path)}")

    # ---- provenance manifests for every new output (match project convention) ----
    inputs_block = [
        {
            "path": R.relpath(COORDS_NPY),
            "hash": summary["source"]["coords_sha256"],
            "format": "numpy",
            "role": "saved 2-D UMAP visualisation coordinates (float32, 902144x2), read-only",
            "rows": N,
            "cols": 2,
        },
        {
            "path": R.relpath(LABELS_PARQUET),
            "hash": summary["source"]["labels_sha256"],
            "format": "parquet",
            "role": "saved HDBSCAN labels (-1 = noise) keyed to row/doc_key, read-only",
            "rows": N,
            "cols": lt.num_columns,
        },
    ]
    params = {
        "record_of": "improved 2-D UMAP views (re-visualisation only; no UMAP/"
        "HDBSCAN/embedding recomputation)",
        "renderer": "matplotlib hexbin density, viridis, LogNorm colour scale",
        "viewports": {
            "central_99_9_percentiles": ["0.05%", "99.95%"],
            "central_99_percentiles": ["0.5%", "99.5%"],
        },
        "note": "See figures/improved_umap_views/umap_view_summary.json for "
        "quantiles, viewport bounds, and in/out document counts.",
    }
    git = summary["git_commit"]
    script_path = os.path.abspath(__file__)
    for p, note in written:
        R.write_manifest(
            p,
            inputs_block,
            script_path,
            params,
            git,
            out_format="image/png",
            notes=note,
            seed=None,
        )
    R.write_manifest(
        json_path,
        inputs_block,
        script_path,
        params,
        git,
        out_format="other",
        notes="Quantiles, viewport bounds, and inside/outside document "
        "counts for the improved 2-D UMAP views.",
        seed=None,
    )
    print(f"\nwrote {len(written) + 1} manifests")
    print("DONE")

    # machine-readable recap for the run summary
    print("\n=== RECAP ===")
    print(
        json.dumps(
            {
                "quantiles": {"umap_1": q1, "umap_2": q2},
                "viewports": summary["viewports"],
                "extreme_outside_central_99_9": extreme,
                "figures": summary["figures"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
