import re

def area_mm2_from_rpt(text: str) -> float:
    m = re.search(r"Total cell area:\s+([\d.]+)", text)
    if not m:
        raise ValueError("Total cell area not found")
    return float(m.group(1)) / 1e6
