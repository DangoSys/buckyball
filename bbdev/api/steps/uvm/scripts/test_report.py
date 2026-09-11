from pathlib import Path


def discover_tests(bbdir: str, chip: str, ball: str) -> list[Path]:
    root = Path(bbdir) / "bb-tests" / "output" / chip / "workloads" / "src" / "CTest" / "chips" / chip / "balls" / chip / ball / "ctests"
    return sorted(root.glob("*-baremetal"))


def write_test_index(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.write_text(
        "ball_dir bemu verilator\n" + "\n".join(f"{ball} {bemu} {verilator}" for ball, bemu, verilator in rows) + "\n"
    )

