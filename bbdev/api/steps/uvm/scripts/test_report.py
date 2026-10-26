from pathlib import Path


def discover_tests(bbdir: str, design: str, ball: str) -> list[Path]:
    root = (
        Path(bbdir)
        / "bb-tests"
        / "output"
        / design
        / "workloads"
        / "src"
        / "CTest"
        / "designs"
        / design
        / "balls"
        / design
        / ball
        / "ctests"
    )
    return sorted(root.glob("*-baremetal"))


def write_test_index(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.write_text(
        "ball_dir bemu verilator\n"
        + "\n".join(f"{ball} {bemu} {verilator}" for ball, bemu, verilator in rows)
        + "\n"
    )
