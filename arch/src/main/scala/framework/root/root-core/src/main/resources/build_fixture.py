from pathlib import Path
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("destination")
parser.add_argument(
    "fixture",
    choices=("core", "supervisor", "multicore", "irq", "admission", "fpu"),
    nargs="?",
    default="core",
)
options = parser.parse_args()
output = Path(options.destination)
fixture = options.fixture
output.mkdir(parents=True, exist_ok=True)
source = Path(__file__).resolve().parent
subprocess.run(
    [
        "riscv64-none-elf-gcc",
        (
            "-march=rv64imafdc_zicsr_zifencei"
            if fixture == "fpu"
            else "-march=rv64imac_zicsr_zifencei"
        ),
        "-mabi=lp64",
        "-nostdlib",
        "-nostartfiles",
        "-Wl,--build-id=none",
        "-Wl,--no-relax",
        "-T",
        str(source / f"{fixture}.ld"),
        str(source / f"{fixture}.S"),
        "-o",
        str(output / f"{fixture}.elf"),
    ],
    check=True,
)
subprocess.run(
    [
        "riscv64-none-elf-objcopy",
        "-O",
        "verilog",
        str(output / f"{fixture}.elf"),
        str(output / f"{fixture}.hex"),
    ],
    check=True,
)
image = output / f"{fixture}.hex"
image.write_bytes(image.read_bytes().replace(b"\r\n", b"\n"))
disassembly = subprocess.run(
    [
        "riscv64-none-elf-objdump",
        "-d",
        str(output / f"{fixture}.elf"),
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout
(output / f"{fixture}.dis").write_text(disassembly)
import re

exits = re.findall(r"^\s*([0-9a-f]+):[^\n]*\bsd\s+zero,0\(s0\)", disassembly, re.M)
if len(exits) != 1:
    raise RuntimeError(f"Expected one successful exit SD in firmware, found {exits}")
definitions = [f"`define CORE_EXIT_PC 64'h{exits[0]}"]
if fixture in ("supervisor", "irq", "admission", "fpu"):
    symbols = subprocess.run(
        [
            "riscv64-none-elf-nm",
            "--defined-only",
            str(output / f"{fixture}.elf"),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    addresses = {
        name: int(address, 16)
        for address, _, name in (line.split() for line in symbols.splitlines())
    }
if fixture == "irq":
    for name in ("irq_load", "irq_amo", "irq_store"):
        definitions.append(f"`define {name.upper()}_PC 64'h{addresses[name]:x}")
if fixture == "fpu":
    definitions.append(f"`define FPU_ILLEGAL_PC 64'h{addresses['fpu_illegal']:x}")
if fixture == "admission":
    for name, address in sorted(addresses.items()):
        if name.startswith("admit_"):
            definitions.append(f"`define {name.upper()}_PC 64'h{address:x}")
if fixture == "supervisor":
    traps = [
        ("trap_boundary", 1, 0x4000A008),
        ("trap_store", 15, 0x40008000),
        ("trap_sum", 13, 0x40006000),
        ("trap_user_load", 13, 0x40008000),
        ("trap_user_ecall", 8, 0),
        ("trap_supervisor_ecall", 9, 0),
    ]
    definitions.append(f"`define SUPERVISOR_TRAP_COUNT {len(traps)}")
    for index, (name, cause, value) in enumerate(traps):
        definitions += [
            f"`define SUPERVISOR_TRAP_{index}_PC 64'h{addresses[name] - 0x40000000:x}",
            f"`define SUPERVISOR_TRAP_{index}_CAUSE 64'd{cause}",
            f"`define SUPERVISOR_TRAP_{index}_VALUE 64'h{value:x}",
        ]
(output / f"{fixture}_fixture.svh").write_text("\n".join(definitions) + "\n")
