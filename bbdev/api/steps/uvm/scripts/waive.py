import re
from pathlib import Path


def apply_waivers(rtl_dir: Path) -> int:
    if not rtl_dir.is_dir():
        raise ValueError(f"missing generated RTL directory: {rtl_dir}")
    changed = 0
    for path in sorted(rtl_dir.glob("*.sv")):
        lines = path.read_text().splitlines(keepends=True)
        output = []
        for line in lines:
            if re.search(r"^\s*(?:wire|reg|logic)\b.*\b_(?:GEN|T)(?:_\d+)?\b", line) and (
                not output or output[-1] != "//VCS coverage off\n"
            ):
                output.extend(("//VCS coverage off\n", line, "//VCS coverage on\n"))
                changed += 1
            else:
                output.append(line)
        path.write_text("".join(output))
    return changed
