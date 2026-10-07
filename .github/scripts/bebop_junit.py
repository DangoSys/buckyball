import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

artifacts = Path(sys.argv[1])
output = Path(sys.argv[2])
results = [
    json.loads(path.read_text()) for path in sorted(artifacts.glob("*/summary.json"))
]
if not results:
    raise RuntimeError(f"No bebop test results under {artifacts}")

suite = ET.Element("testsuite", name="bebop", tests=str(len(results)))
failures = errors = 0
for result in results:
    case = ET.SubElement(
        suite,
        "testcase",
        classname=result["backend"],
        name=result["workload_name"],
        time=result["elapsed_sec"].removesuffix("s"),
    )
    if result["status"] == "pass":
        continue
    if result["status"] == "fail":
        failures += 1
        tag = "failure"
    elif result["status"] in ("timeout", "crash", "infra_error"):
        errors += 1
        tag = "error"
    else:
        raise ValueError(f"Unknown bebop test status: {result['status']}")
    failure = ET.SubElement(case, tag, message=result["status"])
    failure.text = json.dumps(result, indent=2)

suite.set("failures", str(failures))
suite.set("errors", str(errors))
ET.ElementTree(suite).write(output, encoding="utf-8", xml_declaration=True)
