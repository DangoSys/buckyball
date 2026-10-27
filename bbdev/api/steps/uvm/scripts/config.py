import json
import sys
from pathlib import Path
from google.protobuf.json_format import MessageToDict


def profiles(repo, design, selection, ip):
    repo = Path(repo)
    sys.path.insert(0, str(repo / "bbdev/api/steps/config/scripts"))
    import design_pb2

    if design is None:
        from verification import default_ip_config

        path = default_ip_config(repo, ip)
    else:
        path = (
            repo
            / "examples/designs"
            / design
            / "output/sims"
            / selection
            / "generated/design.pb"
        )
    config = design_pb2.Design.FromString(path.read_bytes())
    entries = decode_profiles(config)
    selected = [entry for entry in entries if ip is None or entry["ip"] == ip]
    if not selected:
        raise ValueError(
            f"No IP verification registrations for design={design!r} ip={ip!r}"
        )
    result = []
    for entry in selected:
        tests = [
            test
            for test in entry["tests"]
            if not test["designs"] or design in test["designs"]
        ]
        if not tests:
            raise ValueError(f"No applicable tests: {entry['ip']}/{entry['profile']}")
        result.append({**entry, "tests": tests})
    return result


def decode_profiles(config):
    entries = []
    for descriptor in config.verification.profiles:
        entry = MessageToDict(
            descriptor,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )
        entry["parameters"] = json.loads(entry.pop("parameters_json"))
        for test in entry["tests"]:
            test["timeout_ns"] = int(test["timeout_ns"])
        entries.append(entry)
    return entries
