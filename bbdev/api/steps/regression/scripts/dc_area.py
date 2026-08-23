import re

_AREA = re.compile(r"Total cell area:\s+([\d.]+)")


def area_mm2_from_rpt(text: str) -> float:
    m = _AREA.search(text)
    if not m:
        raise ValueError("Total cell area not found")
    return float(m.group(1)) / 1e6
