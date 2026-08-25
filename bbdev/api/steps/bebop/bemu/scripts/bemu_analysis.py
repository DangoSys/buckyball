#!/usr/bin/env python3
"""Summarize bebop-bemu itrace/mtrace in bdb.ndjson."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ISA = {
    0: "fence",
    1: "barrier",
    16: "mvout",
    32: "mset",
    33: "mvin",
    34: "mmio_set",
    35: "mvin_mmio",
}

MATRIX_MNEMONICS = frozenset({"MATRIX", "MATRIX_F32"})

_API = Path(__file__).resolve().parents[4]
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))


def abs_log_dir(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("log-dir must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"log-dir must be an absolute path: {value}")
    return path


def _hex_u(value: object, field: str, line_no: int) -> int:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"bdb.ndjson:{line_no}: {field} must be a 0x hex string")
    return int(value, 16)


def chip_maps(bbdir: str, chip: str) -> tuple[dict[int, str], set[int], int]:
    path = (
        Path(bbdir)
        / "examples"
        / "chips"
        / chip
        / "configs"
        / "generated"
        / "config"
        / "config.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    cores = []
    for tile in data["designs"]["tiles"]:
        cores.extend(tile["cores"])
    if len(cores) != 1:
        # isomorphic tiles share the same core pkg; take first unique balldomain+_file
        seen = []
        uniq = []
        for core in cores:
            key = core["balldomain"]["_file"]
            if key not in seen:
                seen.append(key)
                uniq.append(core)
        if len(uniq) != 1:
            raise ValueError(f"chip {chip} has no unique topology core")
        core = uniq[0]
    else:
        core = cores[0]

    ball = core["balldomain"]
    isa = ball["ballISA"]
    ball_path = ball["_file"]

    names: dict[int, str] = dict(ISA)
    matrix: set[int] = set()
    seen: set[int] = set()
    for entry in isa:
        if not isinstance(entry, dict):
            raise ValueError(f"ballISA entry must be a table: {ball_path}")
        funct = entry.get("funct7")
        mnemonic = entry.get("mnemonic")
        if not isinstance(funct, int) or funct < 0:
            raise ValueError(f"ballISA funct7 must be a non-negative int: {ball_path}")
        if not isinstance(mnemonic, str) or not mnemonic:
            raise ValueError(f"ballISA mnemonic must be a non-empty string: {ball_path}")
        if funct in seen:
            raise ValueError(f"duplicate ballISA funct7 {funct}: {ball_path}")
        if funct in ISA:
            raise ValueError(
                f"ballISA funct7 {funct} ({mnemonic}) collides with ISA {ISA[funct]}: {ball_path}"
            )
        seen.add(funct)
        names[funct] = mnemonic.lower()
        if mnemonic in MATRIX_MNEMONICS:
            matrix.add(funct)

    mem = core["memdomain"]
    bank = mem["bank"]
    entries = bank["entries"]
    return names, matrix, entries


def _mnk(rs2: int) -> tuple[int, int, int]:
    return rs2 & 0xFFF, (rs2 >> 12) & 0xFFF, (rs2 >> 24) & 0xFFF


def analysis_dir(
    log_dir: Path,
    *,
    names: dict[int, str],
    matrix: set[int],
    bank_depth: int,
    itrace: bool,
    mtrace: bool,
) -> str:
    if bank_depth <= 0:
        raise ValueError(f"bank_depth must be a positive int, got {bank_depth}")
    if not log_dir.is_dir():
        raise ValueError(f"log-dir is not a directory: {log_dir}")
    bdb = log_dir / "bdb.ndjson"
    if not bdb.is_file():
        raise ValueError(f"missing bdb.ndjson: {bdb}")

    n_itrace = 0
    n_mtrace = 0
    prev_clk: int | None = None
    funct_n: Counter[str] = Counter()
    funct_cyc: Counter[str] = Counter()
    mnk_n: Counter[tuple[int, int, int]] = Counter()
    rows_n: Counter[int] = Counter()
    rows_sum = 0
    rows_1 = 0

    with bdb.open(encoding="utf-8") as source:
        for line_no, raw in enumerate(source, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bdb.ndjson:{line_no}: invalid JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"bdb.ndjson:{line_no}: line must be a JSON object")
            kind = obj.get("type")
            if kind == "itrace":
                n_itrace += 1
                if "clk" not in obj:
                    raise ValueError(f"bdb.ndjson:{line_no}: itrace missing clk")
                clk = obj["clk"]
                if not isinstance(clk, int) or clk < 0:
                    raise ValueError(f"bdb.ndjson:{line_no}: itrace clk must be a non-negative int")
                if prev_clk is not None and clk < prev_clk:
                    raise ValueError(
                        f"bdb.ndjson:{line_no}: itrace clk went backwards {prev_clk} -> {clk}"
                    )
                funct = _hex_u(obj.get("funct"), "funct", line_no)
                if funct not in names:
                    known = ", ".join(f"{v}=0x{k:02x}" for k, v in sorted(names.items()))
                    raise ValueError(
                        f"bdb.ndjson:{line_no}: unknown funct 0x{funct:02x}; known: {known}"
                    )
                name = names[funct]
                delta = clk if prev_clk is None else clk - prev_clk
                funct_n[name] += 1
                funct_cyc[name] += delta
                prev_clk = clk
                if funct in matrix:
                    rs2 = _hex_u(obj.get("rs2"), "rs2", line_no)
                    mnk_n[_mnk(rs2)] += 1
            elif kind == "mtrace":
                n_mtrace += 1
                rows = obj.get("rows")
                if not isinstance(rows, int) or rows < 0:
                    raise ValueError(f"bdb.ndjson:{line_no}: mtrace rows must be a non-negative int")
                rows_n[rows] += 1
                rows_sum += rows
                if rows == 1:
                    rows_1 += 1
            elif kind is None:
                raise ValueError(f"bdb.ndjson:{line_no}: missing type")

    if itrace and n_itrace == 0:
        raise ValueError(f"no itrace events in {bdb}")
    if mtrace and n_mtrace == 0:
        raise ValueError(f"no mtrace events in {bdb}")

    lines: list[str] = [f"log_dir: {log_dir}"]
    if itrace:
        total_cyc = sum(funct_cyc.values())
        if total_cyc == 0:
            raise ValueError(f"itrace span is 0 cycles in {bdb}")
        lines.append("== itrace ==")
        lines.append(f"events: {n_itrace}")
        lines.append(f"span_cycles: {total_cyc}")
        lines.append(f"{'funct':<16} {'n':>10} {'cycles':>14} {'pct':>8}")
        for name, count in funct_n.most_common():
            cyc = funct_cyc[name]
            pct = 100.0 * cyc / total_cyc
            lines.append(f"{name:<16} {count:>10} {cyc:>14} {pct:>7.1f}%")
        lines.append(f"fence: {funct_n.get('fence', 0)}")
        lines.append(f"mset: {funct_n.get('mset', 0)}")
        if matrix:
            lines.append("matrix (M,N,K):")
            if not mnk_n:
                lines.append("  (none)")
            else:
                for (m, n, k), count in mnk_n.most_common():
                    lines.append(f"  ({m},{n},{k})  n={count}")
    if mtrace:
        mean = rows_sum / n_mtrace
        lines.append("== mtrace ==")
        lines.append(f"events: {n_mtrace}")
        lines.append(f"bank_depth: {bank_depth}")
        lines.append(f"mean_rows: {mean:.2f}")
        lines.append(f"mean_rows/bank_depth: {mean / bank_depth:.4f}")
        lines.append(f"rows=1: {rows_1} ({100.0 * rows_1 / n_mtrace:.1f}%)")
        lines.append("rows histogram:")
        for rows, count in sorted(rows_n.items()):
            lines.append(f"  {rows}: {count}")
    return "\n".join(lines) + "\n"


def write_report(log_dir: Path, text: str) -> Path:
    out = log_dir / "analysis.txt"
    out.write_text(text, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--bbdir", type=Path)
    parser.add_argument("--itrace", action="store_true")
    parser.add_argument("--mtrace", action="store_true")
    args = parser.parse_args(argv)
    if not args.itrace and not args.mtrace:
        raise SystemExit("need --itrace and/or --mtrace")
    log_dir = abs_log_dir(str(args.log_dir))
    api_root = Path(__file__).resolve().parents[4]
    if str(api_root) not in sys.path:
        sys.path.insert(0, str(api_root))
    from utils.path import get_buckyball_path

    bbdir = args.bbdir if args.bbdir is not None else Path(get_buckyball_path())
    names, matrix, depth = chip_maps(str(bbdir), args.chip)
    text = analysis_dir(
        log_dir,
        names=names,
        matrix=matrix,
        bank_depth=depth,
        itrace=args.itrace,
        mtrace=args.mtrace,
    )
    write_report(log_dir, text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
