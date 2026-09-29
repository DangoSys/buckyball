import re
from pathlib import Path


def apply_waivers(rtl_dir: Path) -> int:
    if not rtl_dir.is_dir():
        raise ValueError(f"missing generated RTL directory: {rtl_dir}")
    changed = 0
    for path in sorted(rtl_dir.glob("*.sv")):
        lines = path.read_text().splitlines(keepends=True)
        output = []
        excluded_module = False
        for line in lines:
            if line.strip() == "// VCS coverage exclude_file":
                # FIRRTL memory models share a file with the DUT in unsplit RTL.
                output.append("//VCS coverage off\n")
                excluded_module = True
                changed += 1
                continue
            if excluded_module and re.match(r"^\s*endmodule\b", line):
                output.extend((line, "//VCS coverage on\n"))
                excluded_module = False
                continue
            output.append(line)
        path.write_text("".join(output))
    return changed
