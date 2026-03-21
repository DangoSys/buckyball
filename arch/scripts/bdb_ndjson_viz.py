#!/usr/bin/env python3
"""
Parse BDB NDJSON trace and draw a single RoB-state figure.

Only itrace issue->complete intervals are plotted.
Idle gaps (no RoB active) are not shown.

Requires: matplotlib (pip install matplotlib)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bdb_ndjson_annotate import build_funct_map, default_isa_dir, parse_funct_id


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: invalid JSON: {e}") from e
    return rows


def clk_of(rec: dict[str, Any], fallback: int | None) -> int | None:
    if "clk" in rec:
        return int(rec["clk"])
    return fallback


def build_rob_intervals(
    records: list[dict[str, Any]],
) -> tuple[list[tuple[int, int, int, int, str]], list[tuple[int, int, int, int]], bool]:
    """
    Build RoB intervals from itrace events.
    Returns:
      - active windows: issue->complete (t0, t1, domain_id, rob_id, funct_label)
      - in-rob windows: alloc->complete (t0, t1, domain_id, rob_id)
      - used_real_clk
    """
    fmap = build_funct_map(default_isa_dir())
    issue_open: dict[tuple[int, int], tuple[int, str]] = {}
    alloc_open: dict[tuple[int, int], int] = {}
    active_out: list[tuple[int, int, int, int, str]] = []
    inrob_out: list[tuple[int, int, int, int]] = []
    used_clk = False
    seq = 0

    for rec in records:
        if rec.get("type") != "itrace":
            seq += 1
            continue

        ev = str(rec.get("event", ""))
        dom = int(rec.get("domain_id", 0))
        rid = int(rec.get("rob_id", -1))
        if rid < 0:
            seq += 1
            continue
        t = clk_of(rec, seq)
        if t is None:
            t = seq
        else:
            used_clk = True

        key = (dom, rid)
        funct_lbl = ""
        if "funct" in rec:
            try:
                fid = parse_funct_id(rec["funct"])
                funct_lbl = fmap.get(fid, f"0x{fid:x}")
            except SystemExit:
                funct_lbl = str(rec["funct"])

        if ev == "alloc":
            alloc_open[key] = int(t)
        elif ev == "issue":
            issue_open[key] = (int(t), funct_lbl)
        elif ev == "complete":
            if key in issue_open:
                t0, flbl = issue_open.pop(key)
                t1 = int(t)
                if t1 < t0:
                    t0, t1 = t1, t0
                if t1 == t0:
                    t1 += 1
                active_out.append((t0, t1, dom, rid, flbl))
            if key in alloc_open:
                t0 = alloc_open.pop(key)
                t1 = int(t)
                if t1 < t0:
                    t0, t1 = t1, t0
                if t1 == t0:
                    t1 += 1
                inrob_out.append((t0, t1, dom, rid))
        seq += 1

    return active_out, inrob_out, used_clk


def plot_timeline(
    records: list[dict[str, Any]],
    path: Path,
    out_path: Path | None,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required: pip install matplotlib") from e

    if not records:
        raise SystemExit("empty trace")

    rob_itv, inrob_itv, used_clk = build_rob_intervals(records)
    if not rob_itv and not inrob_itv:
        raise SystemExit("no itrace issue/complete pairs found")
    all_t0: list[int] = []
    all_t1: list[int] = []
    if rob_itv:
        all_t0.extend(t0 for t0, _, _, _, _ in rob_itv)
        all_t1.extend(t1 for _, t1, _, _, _ in rob_itv)
    if inrob_itv:
        all_t0.extend(t0 for t0, _, _, _ in inrob_itv)
        all_t1.extend(t1 for _, t1, _, _ in inrob_itv)
    g0 = min(all_t0)
    g1 = max(all_t1)
    if g1 <= g0:
        g1 = g0 + 1

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    rows = sorted(
        {(dom, rid) for _, _, dom, rid, _ in rob_itv}
        | {(dom, rid) for _, _, dom, rid in inrob_itv}
    )
    y_of = {k: i for i, k in enumerate(rows)}

    # Draw in-ROB occupancy (alloc->complete) as light gray background.
    for t0, t1, dom, rid in inrob_itv:
        y = y_of[(dom, rid)]
        ax.barh(
            y,
            width=t1 - t0,
            left=t0,
            height=0.80,
            color="#d9d9d9",
            alpha=0.55,
            edgecolor="none",
            zorder=0,
        )

    for t0, t1, dom, rid, flbl in rob_itv:
        y = y_of[(dom, rid)]
        ax.vlines(
            t0,
            ymin=-0.5,
            ymax=y,
            colors="0.45",
            linestyles="-",
            linewidth=0.6,
            alpha=0.6,
            zorder=1,
        )
        ax.barh(
            y,
            width=t1 - t0,
            left=t0,
            height=0.65,
            color=f"C{dom % 10}",
            alpha=0.9,
            edgecolor="black",
            linewidth=0.3,
            zorder=2,
        )
        if (t1 - t0) > 10 and flbl:
            ax.text(
                (t0 + t1) * 0.5,
                y,
                flbl,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                zorder=3,
            )

    ax.set_xlim(g0, g1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"d{dom}:r{rid}" for dom, rid in rows], fontsize=8)
    ax.set_xlabel("clk" if used_clk else "record order")
    ax.set_ylabel("RoB entry (active only)")
    title_clk = "harness clk" if used_clk else "fallback order index"
    ax.set_title(f"RoB active timeline — {path.name} ({title_clk})")

    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Parse and visualize BDB NDJSON trace")
    p.add_argument("ndjson", type=Path, help="Path to bdb.ndjson")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <input>.timeline.png if not --show-only)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Open interactive window (requires display)",
    )
    args = p.parse_args()

    if not args.ndjson.is_file():
        raise SystemExit(f"not a file: {args.ndjson}")

    records = load_records(args.ndjson)

    out = args.output
    if out is None and not args.show:
        out = args.ndjson.with_suffix(".timeline.png")

    plot_timeline(records, args.ndjson, out, args.show)


if __name__ == "__main__":
    main()
