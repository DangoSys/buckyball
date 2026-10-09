import argparse
from pathlib import Path
import struct
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isa-dir", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    root = source.parents[4]
    support = root / "bb-tests/workloads/lib/bbsw/kernels/rvv"
    sys.path.insert(0, str(support))
    from build_images import array, read_segments

    tools = root / "result/bin"
    args.output.mkdir(parents=True, exist_ok=True)
    objects = []
    for name, path in (
        ("norm_window", source / "norm_window.cpp"),
        ("norm", support / "norm.cpp"),
        ("packmm", source / "packmm.cpp"),
        ("pack", support / "pack.cpp"),
    ):
        obj = args.output / f"{name}.o"
        subprocess.run(
            [
                str(tools / "clang++"),
                "--target=riscv64",
                "-march=rv64imf_zve32f_zvl128b",
                "-mabi=lp64f",
                "-mcmodel=medany",
                "-ffreestanding",
                "-O2",
                "-ffp-contract=off",
                "-ffunction-sections",
                "-fdata-sections",
                f"-I{root / 'bb-tests/workloads/lib'}",
                f"-I{args.isa_dir}",
                "-c",
                str(path),
                "-o",
                str(obj),
            ],
            check=True,
        )
        objects.append(obj)
    for name in ("norm_window", "packmm"):
        elf = args.output / f"{name}.elf"
        subprocess.run(
            [
                str(tools / "riscv64-unknown-elf-ld"),
                "-m",
                "elf64lriscv",
                "--gc-sections",
                "-e",
                name,
                "-T",
                str(support / "kernel.ld"),
                *map(str, objects),
                "-o",
                str(elf),
            ],
            check=True,
        )
        text, data, entry = read_segments(elf)
        image = (
            struct.pack(
                "<6I",
                0x31564B52,
                len(text),
                entry,
                0x40000000,
                len(data),
                0,
            )
            + text
            + data
        )
        (args.output / f"{name}_image.h").write_text(
            "#pragma once\n#include <cstdint>\n"
            + array(f"{name}_image", image)
            + f"constexpr uint32_t {name}_entry = {entry};\n"
            + f"constexpr uint32_t {name}_text = {len(text)};\n"
        )
        print(f"{name}: image={len(image)} text={len(text)} data={len(data)}")


if __name__ == "__main__":
    main()
