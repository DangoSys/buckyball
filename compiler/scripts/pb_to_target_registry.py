#!/usr/bin/env python3
"""Generate the static Buckyball compiler target registry from Chip.pb."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _die(message: str) -> None:
    raise ValueError(message)


def _write(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_proto(repo: Path):
    scripts = repo / "bbdev" / "api" / "steps" / "config" / "scripts"
    if not scripts.is_dir():
        _die(f"missing protobuf bindings directory: {scripts}")
    sys.path.insert(0, str(scripts))
    import chip_pb2  # type: ignore

    return chip_pb2


def _target_name(core) -> str:
    return core.role or core.pkg


def _cxx_string(value: str) -> str:
    if not value:
        _die("empty string is not a valid compiler target field")
    if '"' in value or "\\" in value:
        _die(f"unsupported C++ string in chip.pb: {value!r}")
    return f'"{value}"'


def _profile_core(chip, profile):
    matches = [core for core in chip.cores if _target_name(core) == profile.name]
    if not matches:
        _die(f"profile {profile.name}: no matching CoreInstance")
    first = matches[0]
    expected_isa = [
        (entry.mnemonic, entry.funct7, entry.bid) for entry in first.balldomain.isa
    ]
    expected_balls = [entry.ball_name for entry in first.balldomain.mappings]
    expected_bank = (first.mem.bank.num, first.mem.bank.width, first.mem.bank.entries)
    profile_bank = (profile.bank_num, profile.bank_width, profile.bank_entries)
    if expected_bank != profile_bank:
        _die(
            f"profile {profile.name}: profile bank={profile_bank} disagrees "
            f"with CoreInstance bank={expected_bank}"
        )
    for core in matches[1:]:
        isa = [
            (entry.mnemonic, entry.funct7, entry.bid) for entry in core.balldomain.isa
        ]
        balls = [entry.ball_name for entry in core.balldomain.mappings]
        bank = (core.mem.bank.num, core.mem.bank.width, core.mem.bank.entries)
        if isa != expected_isa or balls != expected_balls or bank != expected_bank:
            _die(f"profile {profile.name}: CoreInstance configuration mismatch")
    return first


def _validate_profile(profile, core) -> None:
    if not profile.name.isidentifier():
        _die(f"profile name is not a C++ identifier: {profile.name!r}")
    has_balls = len(core.balldomain.mappings) > 0
    if has_balls and (
        profile.bank_num == 0 or profile.bank_width == 0 or profile.bank_entries == 0
    ):
        _die(f"profile {profile.name}: bank geometry must be non-zero")

    balls = [entry.ball_name for entry in core.balldomain.mappings]
    if len(set(balls)) != len(balls):
        _die(f"profile {profile.name}: duplicate Ball mapping")

    mnemonics: set[str] = set()
    funct7s: set[int] = set()
    for entry in core.balldomain.isa:
        if not entry.mnemonic.isidentifier():
            _die(f"profile {profile.name}: invalid ISA mnemonic {entry.mnemonic!r}")
        if entry.mnemonic in mnemonics:
            _die(f"profile {profile.name}: duplicate ISA mnemonic {entry.mnemonic}")
        if entry.funct7 in funct7s:
            _die(f"profile {profile.name}: duplicate funct7 {entry.funct7}")
        mnemonics.add(entry.mnemonic)
        funct7s.add(entry.funct7)


def _ball_compilers(chip, repo: Path):
    """Resolve the compiler implementation of every Ball enabled by Chip.pb."""
    result = []
    seen: dict[str, str] = {}
    for profile in chip.profiles:
        core = _profile_core(chip, profile)
        _validate_profile(profile, core)
        for mapping in core.balldomain.mappings:
            ball_name = mapping.ball_name
            ball_dir = mapping.ball_dir
            if not ball_name.isidentifier():
                _die(f"profile {profile.name}: invalid Ball name {ball_name!r}")
            if not ball_dir:
                _die(f"profile {profile.name}: {ball_name} has no ball_dir")
            relative_dir = Path(ball_dir)
            if (
                relative_dir.is_absolute()
                or len(relative_dir.parts) != 1
                or ball_dir in {".", ".."}
            ):
                _die(
                    f"profile {profile.name}: {ball_name} has invalid ball_dir "
                    f"{ball_dir!r}"
                )
            previous_dir = seen.get(ball_name)
            if previous_dir is not None:
                if previous_dir != ball_dir:
                    _die(
                        f"Ball {ball_name} maps to both {previous_dir!r} and "
                        f"{ball_dir!r}"
                    )
                continue

            source_dir = repo / "examples" / "balls" / relative_dir / "compiler" / "src"
            dialect_dir = source_dir / "Dialect" / "Buckyball"
            td = dialect_dir / f"{ball_name}.td"
            legalize = dialect_dir / "Transforms" / "LegalizeForLLVMExport.cpp"
            if not td.is_file():
                _die(f"Ball {ball_name}: missing dialect definition {td}")
            if not legalize.is_file():
                _die(f"Ball {ball_name}: missing LLVM export lowering {legalize}")

            assign = (
                source_dir
                / "Conversion"
                / "LowerBuckyball"
                / "AssignPhysicalBankPatterns.cpp"
            )
            bank_ssa = (
                source_dir
                / "Conversion"
                / "LowerBuckyball"
                / "LowerBuckyballToBankSSAPatterns.cpp"
            )
            tile_dir = source_dir / "Conversion" / "LowerTileToBuckyball"
            result.append(
                {
                    "name": ball_name,
                    "dialect_dir": dialect_dir,
                    "td": td,
                    "legalize": legalize,
                    "assign": assign if assign.is_file() else None,
                    "bank_ssa": bank_ssa if bank_ssa.is_file() else None,
                    "tile_sources": (
                        sorted(tile_dir.glob("*.cpp")) if tile_dir.is_dir() else []
                    ),
                }
            )
            seen[ball_name] = ball_dir
    if not result:
        _die("Chip.pb enables no Balls")
    return result


def _core_compilers(chip, repo: Path):
    """Resolve optional Core-level composite lowering implementations."""
    result = []
    seen: set[str] = set()
    for profile in chip.profiles:
        core = _profile_core(chip, profile)
        core_name = core.pkg
        if not core_name.isidentifier():
            _die(f"profile {profile.name}: invalid core package {core_name!r}")
        if core_name in seen:
            continue
        source_dir = (
            repo / "examples" / "cores" / core_name / "compiler" / "src" / "Conversion"
        )
        tile = source_dir / "LowerTileToBuckyball" / "CoreTileLowering.cpp"
        bank_dir = source_dir / "LowerBuckyball"
        bank = bank_dir / "CoreBankSSALowering.cpp"
        if tile.is_file() != bank.is_file():
            _die(
                f"Core {core_name}: tile and bank-SSA lowerings must be provided together"
            )
        if tile.is_file():
            result.append(
                {
                    "name": core_name,
                    "tile": tile,
                    "bank_ssa": [bank, *sorted(bank_dir.glob("*Patterns.cpp"))],
                }
            )
        seen.add(core_name)
    return result


def _emit_td(chip, repo: Path) -> str:
    lines = ["// Generated from Chip.pb. Do not edit.", 'include "Buckyball.td"']
    for ball in _ball_compilers(chip, repo):
        lines.append(f'include "{ball["td"].name}"')
    return "\n".join(lines) + "\n"


def _emit_lowering_hooks(chip, repo: Path) -> str:
    balls = _ball_compilers(chip, repo)
    cores = _core_compilers(chip, repo)
    lines = ["// Generated from Chip.pb. Do not edit.", ""]
    lines.append("#ifdef BUCKYBALL_LEGALIZE_HOOK")
    lines.extend(f"BUCKYBALL_LEGALIZE_HOOK({ball['name']})" for ball in balls)
    lines.extend(["#endif", "", "#ifdef BUCKYBALL_ASSIGN_HOOK"])
    lines.extend(
        f"BUCKYBALL_ASSIGN_HOOK({ball['name']})"
        for ball in balls
        if ball["assign"] is not None
    )
    lines.extend(["#endif", "", "#ifdef BUCKYBALL_TILE_HOOK"])
    lines.extend(
        f"BUCKYBALL_TILE_HOOK({ball['name']})" for ball in balls if ball["tile_sources"]
    )
    lines.extend(["#endif", "", "#ifdef BUCKYBALL_BANK_SSA_HOOK"])
    lines.extend(
        f"BUCKYBALL_BANK_SSA_HOOK({ball['name']})"
        for ball in balls
        if ball["bank_ssa"] is not None
    )
    lines.extend(["#endif", "", "#ifdef BUCKYBALL_CORE_TILE_HOOK"])
    lines.extend(
        f'BUCKYBALL_CORE_TILE_HOOK({core["name"].capitalize()}, "{core["name"]}")'
        for core in cores
    )
    lines.extend(["#endif", "", "#ifdef BUCKYBALL_CORE_BANK_SSA_HOOK"])
    lines.extend(
        f'BUCKYBALL_CORE_BANK_SSA_HOOK({core["name"].capitalize()}, "{core["name"]}")'
        for core in cores
    )
    lines.extend(["#endif", ""])
    return "\n".join(lines)


def _print_ball_paths(chip, repo: Path, kind: str) -> None:
    balls = _ball_compilers(chip, repo)
    if kind == "dialect-dir":
        paths = [ball["dialect_dir"] for ball in balls]
    elif kind == "legalize":
        paths = [ball["legalize"] for ball in balls]
    elif kind == "assign":
        paths = [ball["assign"] for ball in balls if ball["assign"] is not None]
    elif kind == "bank-ssa":
        paths = [ball["bank_ssa"] for ball in balls if ball["bank_ssa"] is not None]
    elif kind == "tile":
        paths = [source for ball in balls for source in ball["tile_sources"]]
    else:
        _die(f"unsupported Ball compiler path kind: {kind}")
    for path in paths:
        print(path)


def _print_core_paths(chip, repo: Path, kind: str) -> None:
    cores = _core_compilers(chip, repo)
    if kind == "tile":
        paths = [core["tile"] for core in cores]
    elif kind == "bank-ssa":
        paths = [source for core in cores for source in core["bank_ssa"]]
    else:
        _die(f"unsupported Core compiler path kind: {kind}")
    for path in paths:
        print(path)


def _emit(chip, target: str | None = None) -> str:
    profiles = [
        profile for profile in chip.profiles if target is None or profile.name == target
    ]
    if not profiles:
        if target is None:
            _die("Chip.pb has no compiler profiles")
        _die(f"Chip.pb has no compiler profile {target!r}")
    names: set[str] = set()
    chunks = ["// Generated from Chip.pb. Do not edit.\n"]
    targets: list[str] = []
    for profile in profiles:
        if profile.name in names:
            _die(f"duplicate compiler profile {profile.name}")
        names.add(profile.name)
        core = _profile_core(chip, profile)
        _validate_profile(profile, core)
        stem = profile.name
        chunks.append(f"static const llvm::StringRef k{stem}Balls[] = {{")
        chunks.extend(
            f"  {_cxx_string(entry.ball_name)}," for entry in core.balldomain.mappings
        )
        chunks.append("};")
        chunks.append(
            f"static const buckyball_target::BuckyballBallMapping k{stem}BallMappings[] = {{"
        )
        chunks.extend(
            f"  {{{_cxx_string(entry.ball_name)}, {entry.in_bw}, {entry.out_bw}}},"
            for entry in core.balldomain.mappings
        )
        chunks.append("};")
        chunks.append(
            f"static const buckyball_target::BuckyballIsaEntry k{stem}Isa[] = {{"
        )
        chunks.extend(
            f"  {{{_cxx_string(entry.mnemonic)}, {entry.funct7}}},"
            for entry in core.balldomain.isa
        )
        chunks.append("};\n")
        targets.append(
            "  {"
            f"{_cxx_string(profile.name)}, {_cxx_string(core.pkg)}, {profile.bank_num}, "
            f"{profile.bank_width}, {profile.bank_entries}, "
            f"llvm::ArrayRef(k{stem}Balls), llvm::ArrayRef(k{stem}BallMappings), "
            f"llvm::ArrayRef(k{stem}Isa)"
            "},"
        )
    chunks.append(
        "static const buckyball_target::BuckyballTargetConfig kBuckyballTargets[] = {"
    )
    chunks.extend(targets)
    chunks.append("};\n")
    return "\n".join(chunks)


def _emit_isa_headers(chip, isa_dir: Path) -> None:
    """Emit one C ISA header per compiler target.

    A header describes one core ISA only.  Deliberately do not create a merged
    chip header: funct7 is target-local and may overlap across profiles.
    """
    for profile in chip.profiles:
        core = _profile_core(chip, profile)
        _validate_profile(profile, core)
        lines = ["#ifndef BALL_ISA_H", "#define BALL_ISA_H", ""]
        lines.extend(
            f"#define BB_FUNC7_{entry.mnemonic} {entry.funct7}"
            for entry in core.balldomain.isa
        )
        lines.extend(["", "#endif", ""])
        header = isa_dir / profile.name / "ballISA.h"
        _write(header, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--chip-pb", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--td-out", type=Path)
    parser.add_argument("--lowering-hooks-out", type=Path)
    parser.add_argument("--isa-dir", type=Path)
    parser.add_argument("--target")
    parser.add_argument(
        "--print-ball-compiler-paths",
        choices=("dialect-dir", "legalize", "assign", "bank-ssa", "tile"),
    )
    parser.add_argument("--print-core-compiler-paths", choices=("bank-ssa", "tile"))
    parser.add_argument("--print-targets", action="store_true")
    parser.add_argument("--print-target-balls", action="store_true")
    parser.add_argument("--print-core-targets", action="store_true")
    args = parser.parse_args()

    if (
        args.out is None
        and args.td_out is None
        and args.lowering_hooks_out is None
        and args.isa_dir is None
        and args.print_ball_compiler_paths is None
        and args.print_core_compiler_paths is None
        and not args.print_targets
        and not args.print_target_balls
        and not args.print_core_targets
    ):
        _die("one output mode is required")

    repo = args.repo.resolve()
    if not repo.is_dir():
        _die(f"repository does not exist: {repo}")
    pb_path = args.chip_pb.resolve()
    if not pb_path.is_file():
        _die(f"missing Chip.pb: {pb_path}")
    pb = _load_proto(repo)
    chip = pb.Chip()
    chip.ParseFromString(pb_path.read_bytes())
    if args.out:
        _write(args.out, _emit(chip, args.target))
    if args.td_out:
        _write(args.td_out, _emit_td(chip, repo))
    if args.lowering_hooks_out:
        _write(args.lowering_hooks_out, _emit_lowering_hooks(chip, repo))
    if args.isa_dir:
        _emit_isa_headers(chip, args.isa_dir)
    if args.print_targets:
        for profile in chip.profiles:
            print(profile.name)
    if args.print_target_balls:
        for profile in chip.profiles:
            core = _profile_core(chip, profile)
            _validate_profile(profile, core)
            balls = ",".join(entry.ball_name for entry in core.balldomain.mappings)
            print(f"{profile.name}:{balls}")
    if args.print_core_targets:
        targets = {profile.name for profile in chip.profiles}
        for core in chip.cores:
            target = _target_name(core)
            if target not in targets:
                _die(f"CoreInstance {core.index}: no compiler profile {target}")
            print(f"{core.index}:{target}")
    if args.print_ball_compiler_paths:
        _print_ball_paths(chip, repo, args.print_ball_compiler_paths)
    if args.print_core_compiler_paths:
        _print_core_paths(chip, repo, args.print_core_compiler_paths)


if __name__ == "__main__":
    main()
