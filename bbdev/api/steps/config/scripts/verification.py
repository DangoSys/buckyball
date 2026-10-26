import importlib.util
import json
import tomllib
from pathlib import Path
from google.protobuf.json_format import ParseDict
import design_pb2


def ip_registrations(repo, entries):
    repo = Path(repo)
    catalog = tomllib.loads((repo / "arch/uvm.toml").read_text())
    profiles = []
    identities = set()
    for entry in entries:
        if set(entry) != {"ip", "config"}:
            raise ValueError("UVM registration must contain exactly ip and config")
        ip = entry["ip"]
        source = Path(entry["config"]["_file"]).relative_to(repo)
        identity = (ip, source.stem)
        if identity in identities:
            raise ValueError(f"Duplicate UVM registration: {ip}/{source.stem}")
        identities.add(identity)
        root = repo / catalog["paths"][ip]
        suite = tomllib.loads((root / "src/main/resources/suite.toml").read_text())
        tests = suite["tests"]
        names = [test["name"] for test in tests]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Suite must declare unique test names: {root}")
        for test in tests:
            if test["kind"] not in {"functional", "assertion"}:
                raise ValueError(f"Invalid test kind: {test['kind']}")
            if test["kind"] == "assertion" and not test["assertion"]:
                raise ValueError(
                    f"Assertion test requires an exact assertion message: {test['name']}"
                )
            if test["timeout_ns"] <= 0 or test["wall_timeout_s"] <= 0:
                raise ValueError(
                    f"Positive simulation and wall timeouts are required: {test['name']}"
                )
            if not test["rtl_scopes"] or not set(test["metrics"]) <= {
                "line",
                "cond",
                "tgl",
            }:
                raise ValueError(
                    f"Invalid DUT coverage scope or metrics: {test['name']}"
                )
            if not (root / "src/main/resources" / test["filelist"]).is_file():
                raise FileNotFoundError(test["filelist"])
        config = entry["config"]
        parameters = {
            key: value for key, value in config["parameters"].items() if key != "_file"
        }
        defines = {
            key: parameters[value]
            for key, value in config["defines"].items()
            if key != "_file"
        }
        profile = design_pb2.VerificationProfile(
            ip=ip,
            profile=source.stem,
            config_path=str(source),
            root=str(root.relative_to(repo)),
            parameters_json=json.dumps(parameters),
            defines={key: str(value) for key, value in defines.items()},
            resources="src/main/resources",
            scope="ip",
        )
        build = catalog["build"][ip]
        profile.build.kind = "mill"
        profile.build.directory = build["directory"]
        profile.build.module = build["module"]
        profile.build.arguments.extend(build["arguments"])
        ParseDict(suite["model"], profile.model)
        for test in tests:
            ParseDict(test, profile.tests.add())
        profiles.append(profile)
    return profiles


def default_ip_config(repo, ip):
    repo = Path(repo)
    catalog = tomllib.loads((repo / "arch/uvm.toml").read_text())
    root = repo / catalog["paths"][ip]
    spec = importlib.util.spec_from_file_location(
        "verification_toml", Path(__file__).with_name("1_toml2json.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expanded = module.toml2json(root / "configs/uvm.toml")
    profiles = ip_registrations(repo, [{"ip": ip, "config": expanded}])

    config = design_pb2.Design(name=ip)
    config.verification.ips.append(ip)
    config.verification.profiles.extend(profiles)
    output = root / "build/verification/default/design.pb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(config.SerializeToString(deterministic=True))
    return output
