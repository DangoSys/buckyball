import re
from pathlib import Path


WAIVE_LINE_PATTERNS = (
    re.compile(r"^\s*(?:wire|reg|logic)\b.*\b_(?:GEN|T)(?:_\d+)?\b"),
)


def apply_waivers(rtl_dir: Path) -> int:
    if not rtl_dir.is_dir():
        raise ValueError(f"missing generated RTL directory: {rtl_dir}")
    changed = 0
    for path in sorted(rtl_dir.glob("*.sv")):
        lines = path.read_text().splitlines(keepends=True)
        output = []
        for line in lines:
            if any(pattern.search(line) for pattern in WAIVE_LINE_PATTERNS) and (
                not output or output[-1] != "//VCS coverage off\n"
            ):
                output.extend(("//VCS coverage off\n", line, "//VCS coverage on\n"))
                changed += 1
            else:
                output.append(line)
        path.write_text("".join(output))
    return changed
