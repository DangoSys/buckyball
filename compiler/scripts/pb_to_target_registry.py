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


def _core_compiler_dirs(chip, repo: Path) -> list[Path]:
    """Resolve the Core compiler packages selected by Chip.pb."""
    result: list[Path] = []
    seen: set[str] = set()
    for profile in chip.profiles:
        core = _profile_core(chip, profile)
        core_name = core.pkg
        if not core_name.isidentifier():
            _die(f"profile {profile.name}: invalid core package {core_name!r}")
        if core_name in seen:
            continue
        compiler_dir = repo / "examples" / "cores" / core_name / "compiler"
        if not (compiler_dir / "CMakeLists.txt").is_file():
            _die(f"Core {core_name}: missing compiler package {compiler_dir}")
        result.append(compiler_dir)
        seen.add(core_name)
    return result


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
        mappings = list(core.balldomain.mappings)
        isa = list(core.balldomain.isa)
        if mappings:
            chunks.append(f"static const llvm::StringRef k{stem}Balls[] = {{")
            chunks.extend(f"  {_cxx_string(entry.ball_name)}," for entry in mappings)
            chunks.append("};")
            chunks.append(
                f"static const buckyball_target::BuckyballBallMapping k{stem}BallMappings[] = {{"
            )
            chunks.extend(
                f"  {{{_cxx_string(entry.ball_name)}, {entry.in_bw}, {entry.out_bw}}},"
                for entry in mappings
            )
            chunks.append("};")
            balls_ref = f"llvm::ArrayRef(k{stem}Balls)"
            mappings_ref = f"llvm::ArrayRef(k{stem}BallMappings)"
        else:
            balls_ref = "llvm::ArrayRef<llvm::StringRef>()"
            mappings_ref = "llvm::ArrayRef<buckyball_target::BuckyballBallMapping>()"
        if isa:
            chunks.append(
                f"static const buckyball_target::BuckyballIsaEntry k{stem}Isa[] = {{"
            )
            chunks.extend(
                f"  {{{_cxx_string(entry.mnemonic)}, {entry.funct7}}}," for entry in isa
            )
            chunks.append("};")
            isa_ref = f"llvm::ArrayRef(k{stem}Isa)"
        else:
            isa_ref = "llvm::ArrayRef<buckyball_target::BuckyballIsaEntry>()"
        chunks.append("")
        targets.append(
            "  {"
            f"{_cxx_string(profile.name)}, {_cxx_string(core.pkg)}, {profile.bank_num}, "
            f"{profile.bank_width}, {profile.bank_entries}, "
            f"{balls_ref}, {mappings_ref}, {isa_ref}"
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
    parser.add_argument("--isa-dir", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--print-core-compiler-dirs", action="store_true")
    parser.add_argument("--print-targets", action="store_true")
    parser.add_argument("--print-bank-targets", action="store_true")
    parser.add_argument("--print-target-balls", action="store_true")
    parser.add_argument("--print-core-targets", action="store_true")
    args = parser.parse_args()

    if (
        args.out is None
        and args.isa_dir is None
        and not args.print_core_compiler_dirs
        and not args.print_targets
        and not args.print_bank_targets
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
    if args.isa_dir:
        _emit_isa_headers(chip, args.isa_dir)
    if args.print_targets:
        for profile in chip.profiles:
            print(profile.name)
    if args.print_bank_targets:
        for profile in chip.profiles:
            if profile.bank_num > 0:
                print(profile.name)
    if args.print_target_balls:
        for profile in chip.profiles:
            core = _profile_core(chip, profile)
            _validate_profile(profile, core)
            dirs = []
            seen = set()
            for entry in core.balldomain.mappings:
                ball_dir = entry.ball_dir
                if not ball_dir:
                    _die(f"profile {profile.name}: {entry.ball_name} has no ball_dir")
                if ball_dir in seen:
                    _die(f"profile {profile.name}: duplicate ball_dir {ball_dir!r}")
                seen.add(ball_dir)
                dirs.append(ball_dir)
            print(f"{profile.name}:{','.join(dirs)}")
    if args.print_core_targets:
        targets = {profile.name for profile in chip.profiles}
        for core in chip.cores:
            target = _target_name(core)
            if target not in targets:
                _die(f"CoreInstance {core.index}: no compiler profile {target}")
            print(f"{core.index}:{target}")
    if args.print_core_compiler_dirs:
        for compiler_dir in _core_compiler_dirs(chip, repo):
            print(compiler_dir)


if __name__ == "__main__":
    main()
