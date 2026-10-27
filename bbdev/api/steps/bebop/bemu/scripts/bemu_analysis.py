#!/usr/bin/env python3
"""Summarize bebop-bemu itrace/mtrace in bdb.ndjson."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if str(Path(__file__).resolve().parents[4]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def _hex_u(value: object, field: str, line_no: int) -> int:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"bdb.ndjson:{line_no}: {field} must be a 0x hex string")
    return int(value, 16)


def design_maps(
    bbdir: str, design: str, selection: str
) -> tuple[dict[int, str], set[int], int]:
    system_isa = {
        0: "fence",
        1: "barrier",
        16: "mvout",
        32: "mset",
        33: "mvin",
        34: "mmio_set",
        35: "mvin_mmio",
    }
    from utils.path import load_design_config

    data = load_design_config(bbdir, design, selection=selection)
    configurations = {
        (
            p.thread.balldomain.SerializeToString(deterministic=True),
            p.thread.mem.bank.entries,
        ): p.thread
        for p in data.profiles
    }
    if len(configurations) != 1:
        raise ValueError(
            f"design {design} requires a unique instruction map and bank depth for this analysis"
        )
    config = next(iter(configurations.values()))
    names: dict[int, str] = dict(system_isa)
    matrix: set[int] = set()
    for instance in config.balldomain.balls:
        for instruction in instance.instructions:
            funct, mnemonic = instruction.funct7, instruction.mnemonic
            if funct in names:
                raise ValueError(
                    f"design {design}: duplicate or reserved funct7 {funct}"
                )
            names[funct] = mnemonic.lower()
            if mnemonic in {"MATRIX", "MATRIX_F32", "SMATMUL_OS"}:
                matrix.add(funct)
    return names, matrix, config.mem.bank.entries


def _mnk(rs1: int, rs2: int) -> tuple[int, int, int]:
    return rs2 & 0xFFF, (rs2 >> 12) & 0xFFF, rs1 >> 30


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
        raise ValueError(f"run is not a directory: {log_dir}")
    bdb = log_dir / "bdb.ndjson"
    if not bdb.is_file():
        raise ValueError(f"missing bdb.ndjson: {bdb}")

    n_itrace = 0
    n_mtrace = 0
    prev_event: int | None = None
    funct_n: Counter[str] = Counter()
    mnk_n: Counter[tuple[int, int, int]] = Counter()
    bank_n: Counter[int] = Counter()
    access_n: Counter[str] = Counter()

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
                event_index = obj["event_index"]
                if not isinstance(event_index, int) or event_index < 0:
                    raise ValueError(
                        f"bdb.ndjson:{line_no}: event_index must be a non-negative int"
                    )
                if prev_event is not None and event_index <= prev_event:
                    raise ValueError(
                        f"bdb.ndjson:{line_no}: instruction event order is not increasing"
                    )
                funct = _hex_u(obj.get("funct"), "funct", line_no)
                if funct not in names:
                    known = ", ".join(
                        f"{v}=0x{k:02x}" for k, v in sorted(names.items())
                    )
                    raise ValueError(
                        f"bdb.ndjson:{line_no}: unknown funct 0x{funct:02x}; known: {known}"
                    )
                name = names[funct]
                funct_n[name] += 1
                prev_event = event_index
                if funct in matrix:
                    rs1 = _hex_u(obj.get("rs1"), "rs1", line_no)
                    rs2 = _hex_u(obj.get("rs2"), "rs2", line_no)
                    mnk_n[_mnk(rs1, rs2)] += 1
            elif kind == "mtrace":
                n_mtrace += 1
                bank_n[obj["vbank_id"]] += 1
                access_n[obj["event"]] += 1
            elif kind is None:
                raise ValueError(f"bdb.ndjson:{line_no}: missing type")

    if itrace and n_itrace == 0:
        raise ValueError(f"no itrace events in {bdb}")
    if mtrace and n_mtrace == 0:
        raise ValueError(f"no mtrace events in {bdb}")

    lines: list[str] = [f"log_dir: {log_dir}"]
    if itrace:
        lines.append("== itrace ==")
        lines.append(f"events: {n_itrace}")
        lines.append(f"{'funct':<16} {'n':>10} {'pct':>8}")
        for name, count in funct_n.most_common():
            pct = 100.0 * count / n_itrace
            lines.append(f"{name:<16} {count:>10} {pct:>7.1f}%")
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
        lines.append("== mtrace ==")
        lines.append(f"events: {n_mtrace}")
        lines.append(f"bank_depth: {bank_depth}")
        lines.append(f"reads: {access_n['read']}")
        lines.append(f"writes: {access_n['write']}")
        lines.append("bank events:")
        for bank, count in sorted(bank_n.items()):
            lines.append(f"  {bank}: {count}")
    return "\n".join(lines) + "\n"


def write_report(log_dir: Path, text: str) -> Path:
    out = log_dir / "analysis.txt"
    out.write_text(text, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--bbdir", type=Path)
    parser.add_argument("--itrace", action="store_true")
    parser.add_argument("--mtrace", action="store_true")
    args = parser.parse_args(argv)
    if not args.itrace and not args.mtrace:
        raise SystemExit("need --itrace and/or --mtrace")
    api_root = Path(__file__).resolve().parents[4]
    if str(api_root) not in sys.path:
        sys.path.insert(0, str(api_root))
    from utils.path import get_buckyball_path, run_log_dir

    bbdir = args.bbdir if args.bbdir is not None else Path(get_buckyball_path())
    log_dir = run_log_dir(bbdir, args.run)
    names, matrix, depth = design_maps(str(bbdir), args.design, args.target)
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
