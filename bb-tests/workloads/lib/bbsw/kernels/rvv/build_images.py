import argparse
from pathlib import Path
import struct
import subprocess

KERNELS = {
    "silu": "rvv_silu_launch",
    "swiglu": "rvv_swiglu_launch",
    "snake": "rvv_snake",
    "quant": "rvv_quant",
    "matmul": "rvv_matmul",
    "norm": "rvv_norm_launch",
    "softmax": "rvv_softmax_launch",
    "attention_softmax": "rvv_attention_softmax_launch",
    "rope": "rvv_rope_launch",
    "pack": "rvv_pack",
}


def read_segments(path):
    elf = path.read_bytes()
    header = struct.unpack_from("<16sHHIIIIIHHHHHH", elf)
    ident, kind, machine = header[:3]
    entry, shoff, shsize, shnum, names_index = (
        header[4],
        header[6],
        header[11],
        header[12],
        header[13],
    )
    if ident[:6] != b"\x7fELF\x01\x01" or kind != 2 or machine != 243:
        raise ValueError(f"{path}: expected a linked little-endian RISC-V ELF32")
    sections = [
        struct.unpack_from("<IIIIIIIIII", elf, shoff + index * shsize)
        for index in range(shnum)
    ]
    names_section = sections[names_index]
    names = elf[names_section[4] : names_section[4] + names_section[5]]
    sections = {
        names[section[0] : names.index(0, section[0])].decode(): section
        for section in sections
    }
    for name, section in sections.items():
        if (
            section[2] & 2
            and section[5]
            and (section[2] & 1 or name not in (".text", ".rodata"))
        ):
            raise ValueError(f"{path}: unsupported allocated section {name}")
    text = sections[".text"]
    if (
        text[1] != 1
        or text[3] != 0
        or text[5] > 4096
        or text[5] % 4
        or entry >= text[5]
        or entry % 4
    ):
        raise ValueError(f"{path}: invalid RVV instruction memory layout")
    data_address = 0x80000000
    initialized = b""
    if ".rodata" in sections and sections[".rodata"][5]:
        section = sections[".rodata"]
        if section[1] != 1 or section[3] != data_address or section[5] > 4096:
            raise ValueError(f"{path}: invalid RVV constant memory layout")
        initialized = elf[section[4] : section[4] + section[5]]
    initialized += bytes((-len(initialized)) % 4)
    return elf[text[4] : text[4] + text[5]], initialized, entry


def array(name, content):
    rows = [
        "  " + ", ".join(f"0x{value:02x}" for value in content[offset : offset + 16])
        for offset in range(0, len(content), 16)
    ]
    return f"alignas(4) const uint8_t {name}[] = {{\n" + ",\n".join(rows) + "\n};\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    tools = source.parents[5] / "result/bin"
    args.output.mkdir(parents=True, exist_ok=True)
    objects = []
    for filename in (
        "silu.cpp",
        "snake.cpp",
        "quant.cpp",
        "matmul.cpp",
        "norm.cpp",
        "softmax.cpp",
        "rope.cpp",
        "pack.cpp",
        "math/exp.cpp",
        "math/sin.cpp",
        "math/cos.cpp",
    ):
        output = args.output / (Path(filename).stem + ".o")
        subprocess.run(
            [
                str(tools / "clang++"),
                "--target=riscv32",
                "-march=rv32imafd_zve64d_zvl256b",
                "-mabi=ilp32d",
                "-ffreestanding",
                "-O2",
                "-ffp-contract=off",
                "-ffunction-sections",
                "-fdata-sections",
                "-c",
                str(source / filename),
                "-o",
                str(output),
            ],
            check=True,
        )
        objects.append(output)
    arrays = []
    images = []
    for name, symbol in KERNELS.items():
        path = args.output / (name + ".elf")
        subprocess.run(
            [
                str(tools / "riscv64-unknown-elf-ld"),
                "-m",
                "elf32lriscv",
                "--gc-sections",
                "-e",
                symbol,
                "-T",
                str(source / "kernel.ld"),
                *map(str, objects),
                "-o",
                str(path),
            ],
            check=True,
        )
        text, data, entry = read_segments(path)
        image = (
            struct.pack("<6I", 0x31564B52, len(text), entry, 0x80000000, len(data), 0)
            + text
            + data
        )
        arrays.append(array(name + "_image", image))
        images.append(
            f"const KernelImage {name}{{{name}_image, sizeof({name}_image), "
            f"{entry}, {len(text)}}};"
        )
        print(
            f"{name}: image={len(image)} text={len(text)} data={len(data)} "
            f"bss=0 entry={entry}"
        )
    (args.output / "images.h").write_text(
        "#pragma once\n#include <cstdint>\n\nnamespace images {\n"
        "struct KernelImage {\n"
        "  const uint8_t *bytes;\n  uint32_t size;\n"
        "  uint32_t entry;\n  uint32_t text_bytes;\n};\n"
        + "".join(f"extern const KernelImage {name};\n" for name in KERNELS)
        + "}\n"
    )
    (args.output / "images.cpp").write_text(
        '#include "images.h"\n\nnamespace images {\nnamespace {\n'
        + "\n".join(arrays)
        + "}\n"
        + "\n".join(images)
        + "\n}\n"
    )


if __name__ == "__main__":
    main()
