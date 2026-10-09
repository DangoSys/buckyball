import fcntl
import os
import re
import shlex
import sys
import tomllib
from datetime import datetime
from pathlib import Path

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path, log_dir
from utils.stream_run import stream_run_logger

config_scripts = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config", "scripts")
)
sys.path.insert(0, config_scripts)


def load_chip(bbdir: str, chip: str):
    import chip_pb2

    path = (
        Path(bbdir) / "examples" / "chips" / chip / "configs" / "generated" / "chip.pb"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run bbdev config --install")
    msg = chip_pb2.Chip()
    msg.ParseFromString(path.read_bytes())
    if not msg.name or not msg.cores:
        raise ValueError(f"empty chip.pb: {path}")
    return msg


def load_chip_uvm(bbdir: str, chip: str) -> dict[str, list[str]]:
    path = Path(bbdir) / "examples" / "chips" / chip / "configs" / "chip.toml"
    config = tomllib.loads(path.read_text())["uvm"]
    if set(config) != {"balls", "ips"}:
        raise ValueError(f"{path}: [uvm] must define exactly balls and ips")
    if not config["balls"] and not config["ips"]:
        raise ValueError(f"{path}: [uvm] has no targets")
    return config


def ball_domain(chip, ball: str | None = None):
    if ball is None:
        domains = [core.balldomain for core in chip.cores]
        domain = domains[0]
        for other in domains[1:]:
            if other.mappings != domain.mappings or other.isa != domain.isa:
                raise ValueError("chip.pb cores have different balldomains")
        return domain

    matches = []
    for core in chip.cores:
        mappings = [m for m in core.balldomain.mappings if m.ball_dir == ball]
        if mappings:
            mapping = mappings[0]
            isa = [e for e in core.balldomain.isa if e.bid == mapping.ball_id]
            matches.append((core.balldomain, mapping, isa))
    if not matches:
        raise ValueError(f"ball {ball!r} not in chip.pb")

    domain, mapping, isa = matches[0]
    for _, other_mapping, other_isa in matches[1:]:
        if other_mapping != mapping or other_isa != isa:
            raise ValueError(f"ball {ball!r} has different definitions across cores")
    return domain


def selected_mappings(domain, ball: str | None):
    if ball is None:
        return list(domain.mappings)
    for m in domain.mappings:
        if m.ball_dir == ball:
            return [m]
    raise ValueError(f"ball {ball!r} not in chip.pb")


def vcs_defines(domain, mapping, core, tile, parameters):
    shared_bank_num = (
        tile.shared_mem.entries // core.mem.bank.entries
        if tile.shared_mem.enable
        else 0
    )
    max_group_count = max(core.mem.bank.num, shared_bank_num)
    defs = [
        f"+define+BB_IN_BW={mapping.in_bw}",
        f"+define+BB_OUT_BW={mapping.out_bw}",
        f"+define+BB_MMIO_READ_BW={mapping.mmio_read_bw}",
        f"+define+BB_MMIO_WRITE_BW={mapping.mmio_write_bw}",
        f"+define+BB_BANK_ID_W={(tile.virtual_bank_count - 1).bit_length()}",
        f"+define+BB_GROUP_COUNT_W={max_group_count.bit_length()}",
        f"+define+BB_GROUP_ID_W={(max_group_count - 1).bit_length()}",
        f"+define+BB_BANK_ADDR_W={(core.mem.bank.entries - 1).bit_length()}",
        f"+define+BB_ROB_ID_W={(core.frontend.rob_entries - 1).bit_length()}",
        f"+define+BB_SUB_ROB_ID_W={(core.frontend.sub_rob_depth * 4 - 1).bit_length()}",
    ]
    for e in domain.isa:
        if e.bid == mapping.ball_id:
            defs.append(f"+define+{e.mnemonic}_FUNCT7={e.funct7}")
    for define, parameter in parameters.items():
        defs.append(f"+define+{define}={mapping.ball_params[parameter]}")
    return defs


def _filelist(
    verify_dir: Path,
    ball_dir: str,
    uvm_rel: str,
    rtl_dir: Path,
    sim_dir: Path,
) -> str:
    src = verify_dir / "filelists" / f"{ball_dir}_ball.f"
    dst = sim_dir / f"{ball_dir}_ball.f"
    sim_dir.mkdir(parents=True, exist_ok=True)
    text = src.read_text()
    dst.write_text(text.replace("@UVM@", uvm_rel).replace("@RTL@", str(rtl_dir.resolve())))
    return str(dst.relative_to(verify_dir))


def load_ip(bbdir: str, name: str, target_name: str | None = None, chip: str | None = None) -> dict:
    registry = Path(bbdir) / "arch" / "uvm.toml"
    registered = tomllib.loads(registry.read_text())
    if name not in registered["ips"]:
        raise ValueError(f"IP {name!r} is not registered in {registry}")
    root = Path(bbdir) / registered["paths"][name]
    build = registered.get("build", {}).get(name, {})
    if set(build) - {"directory", "module", "arguments"}:
        raise ValueError(f"Invalid IP build settings for {name!r}: {build}")
    directory = build.get("directory", str(root.relative_to(bbdir)))
    module = build.get("module", name)
    arguments = build.get("arguments", [])
    if not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments):
        raise ValueError(f"IP build arguments must be a list of strings: {name!r}")
    if not isinstance(directory, str) or not directory or not isinstance(module, str) or not module:
        raise ValueError(f"IP build directory and module must be nonempty strings: {name!r}")
    resources = root / "src" / "main" / "resources"
    targets = [
        {
            "name": filelist.stem,
            "filelist": filelist.name,
            "top": f"{filelist.stem}_tb",
            "test": "protocol_test",
        }
        for filelist in sorted(resources.glob("*.f"))
    ]
    for target in targets:
        metadata = resources / f"{target['name']}.toml"
        if metadata.is_file():
            settings = tomllib.loads(metadata.read_text())
            if not settings or set(settings) - {"expected_assertion", "chips", "invalid_case", "rtl_scopes"}:
                raise ValueError(f"Invalid IP test settings: {metadata}")
            if "rtl_scopes" in settings:
                scopes = settings["rtl_scopes"]
                if not isinstance(scopes, list) or not scopes or any(
                    not isinstance(scope, str) or not re.fullmatch(
                        r"[A-Za-z_][A-Za-z0-9_$]*(?:\[\d+\])?(?:\.[A-Za-z_][A-Za-z0-9_$]*(?:\[\d+\])?)*", scope
                    ) for scope in scopes
                ):
                    raise ValueError(f"Invalid RTL instance scopes: {metadata}")
                target["rtl_scopes"] = scopes
            if "invalid_case" in settings:
                case = settings["invalid_case"]
                if type(case) is not int or case < 0 or "expected_assertion" not in settings:
                    raise ValueError(f"invalid_case requires a nonnegative integer and expected_assertion: {metadata}")
                target["invalid_case"] = case
            if "expected_assertion" in settings:
                assertion = settings["expected_assertion"]
                if not isinstance(assertion, str) or not assertion or "\n" in assertion:
                    raise ValueError(f"Expected one assertion message in {metadata}")
                target["expected_assertion"] = assertion
            if "chips" in settings:
                profiles = settings["chips"]
                if (not isinstance(profiles, list) or not profiles or
                    not all(isinstance(value, str) and value.strip() and value == value.strip() for value in profiles)):
                    raise ValueError(f"chips must be a nonempty list of nonempty chip names: {metadata}")
                if chip is None:
                    raise ValueError(f"Chip is required to select declared IP profiles: {metadata}")
                target["chips"] = profiles
    if not targets:
        raise ValueError(f"registered IP {name!r} has no filelists under {resources}")
    if target_name is not None:
        targets = [target for target in targets if target["name"] == target_name]
        if not targets:
            raise ValueError(f"Unknown target {target_name!r} for IP {name!r}")
        if "chips" in targets[0] and chip not in targets[0]["chips"]:
            raise ValueError(f"Target {target_name!r} for IP {name!r} does not support chip {chip!r}; declared chips: {targets[0]['chips']}")
    targets = [target for target in targets if "chips" not in target or chip in target["chips"]]
    if not targets:
        raise ValueError(f"IP {name!r} has no applicable targets for chip {chip!r}")
    return {
        "root": str(root.relative_to(bbdir)),
        "mill_module": module,
        "mill_directory": directory,
        "mill_arguments": arguments,
        "model": "src/csrc/Cargo.toml",
        "resources": "src/main/resources",
        "targets": targets,
    }


def _ip_filelist(
    resources: Path, source: str, sim_dir: Path, bbdir: str, rtl: Path
) -> Path:
    dst = sim_dir / source
    sim_dir.mkdir(parents=True, exist_ok=True)
    text = (resources / source).read_text()
    dst.write_text(
        text.replace("@VERIFY@", str(Path(bbdir) / "verify"))
        .replace("@RESOURCES@", str(resources))
        .replace("@RTL@", str(rtl))
    )
    return dst


def check_uvm_result(result, target: str, expected_assertion: str | None = None) -> None:
    output = result.stdout + result.stderr
    if expected_assertion is not None:
        assertions = re.findall(r"Assertion failed: ([^\r\n]+)", output)
        failure_counts = re.findall(r"UVM_(?:ERROR|FATAL)\s*:\s*(\d+)", output)
        if (
            not assertions
            or set(assertions) != {expected_assertion}
            or "Fatal:" not in output
            or any(int(count) for count in failure_counts)
            or result.returncode < 0
        ):
            raise RuntimeError(f"Expected RTL assertion did not terminate {target}: {expected_assertion}")
        return
    if result.returncode != 0:
        raise RuntimeError(f"UVM failed for {target}")
    counts = dict(
        re.findall(r"UVM_(ERROR|FATAL)\s*:\s*(\d+)", output)
    )
    if set(counts) != {"ERROR", "FATAL"}:
        raise RuntimeError(f"UVM report summary missing for {target}")
    if counts != {"ERROR": "0", "FATAL": "0"}:
        raise RuntimeError(
            f"UVM failed for {target}: errors={counts['ERROR']} fatals={counts['FATAL']}"
        )


def build_ip(bbdir: str, chip: str, name: str, ctx, target_name: str | None = None) -> dict:
    config = load_ip(bbdir, name, target_name, chip)
    root = Path(bbdir) / config["root"]
    resources = root / config["resources"]
    manifest = root / config["model"]
    rtl = root / "build"
    sim_root = rtl / "uvm" / chip
    rtl.mkdir(parents=True, exist_ok=True)
    build_lock = (rtl / ".build.lock").open("w")
    fcntl.flock(build_lock, fcntl.LOCK_EX)

    cargo = (
        f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command "
        f"cargo build --manifest-path {shlex.quote(str(manifest))}"
    )
    result = stream_run_logger(
        cmd=cargo,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="uvm dpi",
        stderr_prefix="uvm dpi",
    )
    if result.returncode != 0:
        raise RuntimeError(f"cargo build failed for IP {name}")

    emit = shlex.join([
        str(Path(bbdir) / "result" / "bin" / "mill"),
        config["mill_module"] + ".run",
        *(argument.replace("{chip}", chip).replace("{targets}", ",".join(target["name"] for target in config["targets"])) for argument in config["mill_arguments"]),
    ])
    result = stream_run_logger(
        cmd=emit,
        logger=ctx.logger,
        cwd=str(Path(bbdir) / config["mill_directory"]),
        stdout_prefix="uvm rtl",
        stderr_prefix="uvm rtl",
    )
    if result.returncode != 0:
        raise RuntimeError(f"RTL generation failed for IP {name}")

    for target in config["targets"]:
        sim_dir = sim_root / target["name"]
        filelist = _ip_filelist(resources, target["filelist"], sim_dir, bbdir, rtl)
        simv = sim_dir / "simv"
        csrc = sim_dir / "csrc"
        hier = sim_dir / "cm_hier.cfg"
        declared_hier = resources / f"{target['name']}.cm_hier"
        if declared_hier.is_file():
            scopes = [line.strip() for line in declared_hier.read_text().splitlines() if line.strip()]
            if not scopes or any(
                not re.fullmatch(r"\+tree @TOP@\.[A-Za-z_][A-Za-z0-9_$]*(?:\[\d+\])?(?:\.[A-Za-z_][A-Za-z0-9_$]*(?:\[\d+\])?)*", line)
                for line in scopes
            ):
                raise ValueError(f"Invalid IP coverage hierarchy: {declared_hier}")
            hier.write_text("\n".join(line.replace("@TOP@", target["top"]) for line in scopes) + "\n")
        else:
            hier.write_text(
                f"+tree {target['top']}.dut\n"
                f"+tree {target['top']}.source_if\n"
                f"+tree {target['top']}.sink_if\n"
            )
        script = (
            f"cd {shlex.quote(str(sim_dir))} && "
            f"rm -rf {shlex.quote(str(csrc))} {shlex.quote(str(simv))} {shlex.quote(str(simv))}.daidir && "
            f"mkdir -p {shlex.quote(str(csrc))} && "
            f"vcs -full64 -sverilog -timescale=1ns/1ps -debug_access+all -top {shlex.quote(target['top'])} "
            "${=VCS_UVM_ARGS} "
            f"-cm line+cond+tgl+assert -cm_hier {shlex.quote(str(hier))} "
            f"-cm_assert_hier {shlex.quote(str(hier))} "
            f"-Mdir={shlex.quote(str(csrc))} -o {shlex.quote(str(simv))} "
            f"-f {shlex.quote(str(filelist))}"
        )
        command = (
            f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command "
            f"zsh -c {shlex.quote(script)}"
        )
        result = stream_run_logger(
            cmd=command,
            logger=ctx.logger,
            cwd=bbdir,
            stdout_prefix="uvm vcs",
            stderr_prefix="uvm vcs",
        )
        if result.returncode != 0:
            raise RuntimeError(f"VCS failed for IP {name} target={target['name']}")
    build_lock.close()
    return config


def validate_ip_exclusions(exclusions: Path, exported: Path) -> None:
    current = set()
    for path in exported.glob("fullexclude.*"):
        for block in path.read_text().split('// CHECKSUM: "')[1:]:
            checksum = block.split('"', 1)[0]
            instance = re.search(r"^// INSTANCE: (\S+)", block, re.MULTILINE)
            if instance:
                current.add((instance[1], checksum))
    checksum = None
    for line in exclusions.read_text().splitlines():
        if line.startswith("CHECKSUM:"):
            checksum = line.split('"')[1]
        elif line.startswith("INSTANCE:"):
            instance = line.split()[1]
            if (instance, checksum) not in current:
                raise RuntimeError(
                    f"coverage exclusions require review: {exclusions}: {instance}"
                )
        elif line.startswith("MODULE:"):
            raise ValueError(f"IP exclusions must use instance scopes: {exclusions}")


def report_ip_coverage(
    bbdir: str, root: Path, target: dict, simv: Path, cov_dir: Path, ctx
) -> None:
    cov_dir.mkdir(parents=True, exist_ok=True)
    exclusions = root / "src" / "main" / "resources" / f"{target['name']}.el"
    hierarchy = simv.parent / "rtl_hier.cfg"
    hierarchy.write_text("".join(
        f"+tree {target['top']}.{instance}\n" for instance in target.get("rtl_scopes", ["dut"])
    ))
    scope = f" -hier {shlex.quote(str(hierarchy))} -metric line+cond+tgl"
    reports = [(cov_dir, ""), (cov_dir / "rtl_raw", scope)]
    if exclusions.is_file():
        reports[0] = (cov_dir, " -dump full_exclusions")
        reports.append((cov_dir / "rtl", scope + f" -elfile {shlex.quote(str(exclusions))} -excl_strict"))
    for report, options in reports:
        if report.name == "rtl":
            validate_ip_exclusions(exclusions, simv.parent)
        command = (
            f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command "
            f"urg -dir {shlex.quote(str(simv))}.vdb -format text"
            f"{options} -report {shlex.quote(str(report))}"
        )
        result = stream_run_logger(
            cmd=command,
            logger=ctx.logger,
            cwd=str(simv.parent),
            stdout_prefix="uvm urg",
            stderr_prefix="uvm urg",
        )
        if result.returncode != 0:
            raise RuntimeError(f"URG failed for {target['name']}: {report}")
        if report.name == "rtl" and re.search(
            r"(?:Warning|Error)-\[", result.stdout + result.stderr
        ):
            raise RuntimeError(
                f"coverage exclusions require review for {target['name']}: {report}"
            )


def run_ip(bbdir: str, chip: str, name: str, ctx, cov_root: str, target_name: str | None = None) -> dict:
    config = build_ip(bbdir, chip, name, ctx, target_name)
    root = Path(bbdir) / config["root"]
    manifest = root / config["model"]
    crate = tomllib.loads(manifest.read_text())["package"]["name"].replace("-", "_")
    model = manifest.parent / "target" / "debug" / f"lib{crate}"
    sim_root = root / "build" / "uvm" / chip

    coverage = []
    expected_assertions = []
    for target in config["targets"]:
        simv = sim_root / target["name"] / "simv"
        simv_q = shlex.quote(str(simv))
        cov_dir = Path(cov_root) / target["name"] / "coverage"
        scenario = f"+invalid_case={target['invalid_case']} " if "invalid_case" in target else ""
        script = (
            f'loader="$(patchelf --print-interpreter {simv_q})"; '
            f'library_path="$(patchelf --print-rpath {simv_q})"; '
            f'"$loader" --library-path "$library_path" {simv_q} -no_save '
            f"-sv_lib {shlex.quote(str(model))} +UVM_TESTNAME={shlex.quote(target['test'])} {scenario}"
            f"-cm line+cond+tgl+assert -cm_name {shlex.quote(target['test'])}"
        )
        command = (
            f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command "
            f"zsh -ic {shlex.quote(script)}"
        )
        result = stream_run_logger(
            cmd=command,
            logger=ctx.logger,
            cwd=str(root),
            stdout_prefix="uvm run",
            stderr_prefix="uvm run",
        )
        expected_assertion = target.get("expected_assertion")
        check_uvm_result(result, f"IP {name} target={target['name']}", expected_assertion)
        if expected_assertion is not None:
            expected_assertions.append(target["name"])
            continue

        report_ip_coverage(bbdir, root, target, simv, cov_dir, ctx)
        coverage.append((target["name"], cov_dir))

    index_dir = Path(cov_root) / "coverage"
    index_dir.mkdir(parents=True, exist_ok=True)
    rows = ["target score line cond toggle assert group result"]
    for target, path in coverage:
        summary = dashboard_summary(str(path))
        rows.append(
            f"{target} {summary['SCORE']} {summary['LINE']} {summary['COND']} "
            f"{summary['TOGGLE']} {summary.get('ASSERT', '--')} {summary.get('GROUP', '--')} pass"
        )
    rows.extend(f"{target} -- -- -- -- -- -- expected-assertion" for target in expected_assertions)
    index = index_dir / "index.txt"
    index.write_text("\n".join(rows) + "\n")
    return {
        "chip": chip,
        "ip": name,
        "targets": [t["name"] for t in config["targets"]],
        "expected_assertions": expected_assertions,
        "index": str(index),
        "rtl_raw_coverage": {
            target["name"]: str(Path(cov_root) / target["name"] / "coverage" / "rtl_raw")
            for target in config["targets"]
            if target["name"] not in expected_assertions
        },
        "rtl_coverage": {
            target["name"]: str(Path(cov_root) / target["name"] / "coverage" / "rtl")
            for target in config["targets"]
            if target["name"] not in expected_assertions
            and (root / "src/main/resources" / f"{target['name']}.el").is_file()
        },
        "log": cov_root,
    }


def build_ball(bbdir: str, chip_name: str, mill_cfg: str, domain, mapping, ctx) -> None:
    ball = mapping.ball_dir
    verify_dir = Path(bbdir) / "examples" / "balls" / ball / "verify"
    casegen = verify_dir / "casegen" / "Cargo.toml"
    rtl_dir = Path(bbdir) / "arch" / "build" / chip_name / mill_cfg
    chip = load_chip(bbdir, chip_name)
    core = next(core for core in chip.cores if any(item.ball_dir == ball for item in core.balldomain.mappings))
    tile = next(tile for tile in chip.tiles if core.index in tile.core_indices)
    sim_dir = verify_dir / "build" / chip_name
    uvm_rel = os.path.relpath(Path(bbdir) / "verify" / "uvm", verify_dir)
    flist = _filelist(verify_dir, ball, uvm_rel, rtl_dir, sim_dir)
    parameters_path = verify_dir / "parameters.toml"
    parameters = tomllib.loads(parameters_path.read_text())["defines"] if parameters_path.is_file() else {}
    cargo = (
        f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command "
        f"cargo build --manifest-path {shlex.quote(str(casegen))}"
    )
    r = stream_run_logger(
        cmd=cargo,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="uvm dpi",
        stderr_prefix="uvm dpi",
    )
    if r.returncode != 0:
        raise RuntimeError(f"cargo build failed for {ball}")
    crate = tomllib.loads(casegen.read_text()).get("package", {}).get("name")
    if not crate:
        raise ValueError(f"package.name missing in {casegen}")
    simv = sim_dir / "simv"
    csrc = sim_dir / "csrc"
    hier = sim_dir / "cm_hier.cfg"
    sim_dir.mkdir(parents=True, exist_ok=True)
    hier.write_text("+tree tb_top.dut\n")
    script = (
        f"cd {shlex.quote(str(verify_dir))} && "
        f"rm -rf {shlex.quote(str(csrc))} {shlex.quote(str(simv))} {shlex.quote(str(simv))}.daidir && "
        f"mkdir -p {shlex.quote(str(sim_dir))} {shlex.quote(str(csrc))} && "
        "vcs -full64 -sverilog -timescale=1ns/1ps -debug_access+all -top tb_top "
        "${=VCS_UVM_ARGS} "
        + " ".join(shlex.quote(d) for d in vcs_defines(domain, mapping, core, tile, parameters))
        + f" -cm line+cond+tgl+assert -cm_hier {shlex.quote(str(hier))} "
        f"-Mdir={shlex.quote(str(csrc))} -o {shlex.quote(str(simv))} "
        f"-f {shlex.quote(flist)}"
    )
    vcs = f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command zsh -c {shlex.quote(script)}"
    r = stream_run_logger(
        cmd=vcs,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="uvm vcs",
        stderr_prefix="uvm vcs",
    )
    if r.returncode != 0:
        raise RuntimeError(f"vcs failed for {ball}")


def run_ball(
    bbdir: str, chip_name: str, mill_cfg: str, domain, mapping, ctx, cov_dir: str
) -> None:
    ball = mapping.ball_dir
    verify_dir = Path(bbdir) / "examples" / "balls" / ball / "verify"
    simv = verify_dir / "build" / chip_name / "simv"
    build_ball(bbdir, chip_name, mill_cfg, domain, mapping, ctx)
    crate = tomllib.loads((verify_dir / "casegen" / "Cargo.toml").read_text())[
        "package"
    ]["name"]
    dpi = verify_dir / "casegen" / "target" / "debug" / f"lib{crate.replace('-', '_')}"
    test = f"{ball}_ball_test"
    verify_config = verify_dir / "build" / chip_name / "verify_config.env"
    chip = load_chip(bbdir, chip_name)
    core = next(core for core in chip.cores if any(item.ball_dir == ball for item in core.balldomain.mappings))
    verify_config.write_text(
        f"bank_entries={core.mem.bank.entries}\nball_id={mapping.ball_id}\n"
    )
    simv_q = shlex.quote(str(simv))
    script = (
        f"cd {shlex.quote(str(verify_dir))} && "
        f'loader="$(patchelf --print-interpreter {simv_q})"; '
        f'library_path="$(patchelf --print-rpath {simv_q})"; '
        f'BB_VERIFY_CONFIG={shlex.quote(str(verify_config))} '
        f'"$loader" --library-path "$library_path" {simv_q} -no_save '
        f"-sv_lib {shlex.quote(str(dpi))} "
        f"+UVM_TESTNAME={shlex.quote(test)} +BID={mapping.ball_id} "
        f"-cm line+cond+tgl+assert -cm_name {shlex.quote(test)}"
    )
    cmd = f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command zsh -ic {shlex.quote(script)}"
    r = stream_run_logger(
        cmd=cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="uvm run",
        stderr_prefix="uvm run",
    )
    check_uvm_result(r, f"{ball} test={test}")
    Path(cov_dir).mkdir(parents=True, exist_ok=True)
    urg = (
        f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command "
        f"urg -dir {shlex.quote(str(simv))}.vdb -format text -report {shlex.quote(cov_dir)}"
    )
    r = stream_run_logger(
        cmd=urg,
        logger=ctx.logger,
        cwd=str(verify_dir),
        stdout_prefix="uvm urg",
        stderr_prefix="uvm urg",
    )
    if r.returncode != 0:
        raise RuntimeError(f"urg failed for {ball}")


def dashboard_summary(cov_dir: str) -> dict[str, str]:
    path = Path(cov_dir) / "dashboard.txt"
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines) and "Total Coverage Summary" not in lines[i]:
        i += 1
    if i >= len(lines):
        raise RuntimeError(f"Total Coverage Summary missing in {path}")
    i += 1
    while i < len(lines) and not lines[i].strip().startswith("SCORE"):
        i += 1
    if i + 1 >= len(lines):
        raise RuntimeError(f"SCORE row missing in {path}")
    keys = lines[i].split()
    vals = lines[i + 1].split()
    if len(vals) != len(keys):
        raise RuntimeError(f"SCORE/value mismatch in {path}: {keys!r} vs {vals!r}")
    return dict(zip(keys, vals))


def run_chip(bbdir: str, chip: str, ball: str | None, ctx, do_run: bool) -> dict:
    msg = load_chip(bbdir, chip)
    domain = ball_domain(msg, ball)
    mill_cfg = msg.mill.verilator_config
    if not mill_cfg:
        raise ValueError(f"chip {chip!r} has no sims.verilator for UVM")
    maps = selected_mappings(domain, ball)
    ran = []
    covs = []
    run_root = None
    if do_run:
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        run_name = "all" if ball is None else maps[0].ball_dir
        run_root = log_dir(bbdir, chip, "verilog", stamp, "uvm", run_name)
    for m in maps:
        if do_run:
            cov = (
                os.path.join(run_root, m.ball_dir, "coverage")
                if ball is None
                else os.path.join(run_root, "coverage")
            )
            run_ball(bbdir, chip, mill_cfg, domain, m, ctx, cov)
            if not (Path(cov) / "dashboard.txt").is_file():
                raise RuntimeError(f"urg wrote no dashboard.txt under {cov}")
            covs.append((m.ball_dir, cov))
        else:
            build_ball(bbdir, chip, mill_cfg, domain, m, ctx)
        ran.append(m.ball_dir)
    info = {"chip": chip, "mill": mill_cfg, "balls": ran}
    if do_run and ball is None:
        index_dir = Path(run_root) / "coverage"
        index_dir.mkdir(parents=True, exist_ok=True)
        body = ["ball_dir score line cond toggle assert group result"]
        for b, p in covs:
            s = dashboard_summary(p)
            body.append(
                f"{b} {s['SCORE']} {s['LINE']} {s['COND']} {s['TOGGLE']} "
                f"{s.get('ASSERT', '--')} {s.get('GROUP', '--')} pass"
            )
        index = index_dir / "index.txt"
        index.write_text("\n".join(body) + "\n")
        info["index"] = str(index)
        info["log"] = run_root
    elif do_run:
        info["log"] = run_root
    return info


def run_uvm(
    bbdir: str, chip: str, ball: str | None, ip: str | None, ctx, do_run: bool, target_name: str | None = None
) -> dict:
    if target_name is not None and ip is None:
        raise ValueError("Parameter --target requires --ip")
    if ball is not None:
        return run_chip(bbdir, chip, ball, ctx, do_run)
    if ip is not None:
        if do_run:
            stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            run_name = f"{ip}-{target_name}" if target_name is not None else ip
            run_root = log_dir(bbdir, chip, "verilog", stamp, "uvm", run_name)
            return run_ip(bbdir, chip, ip, ctx, run_root, target_name)
        config = build_ip(bbdir, chip, ip, ctx, target_name)
        return {
            "chip": chip,
            "ip": ip,
            "targets": [target["name"] for target in config["targets"]],
        }

    targets = load_chip_uvm(bbdir, chip)
    results, failures = [], []
    for ball in targets["balls"]:
        try:
            results.append(run_chip(bbdir, chip, ball, ctx, do_run))
        except Exception as error:
            if not do_run: raise
            ctx.logger.error(f"UVM ball/{ball} failed: {error}")
            failures.append({"target": f"ball/{ball}", "error": str(error)})
    for ip in targets["ips"]:
        try:
            if do_run:
                stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                run_root = log_dir(bbdir, chip, "verilog", stamp, "uvm", ip)
                results.append(run_ip(bbdir, chip, ip, ctx, run_root))
            else:
                config = build_ip(bbdir, chip, ip, ctx)
                results.append({"chip": chip, "ip": ip, "targets": [item["name"] for item in config["targets"]]})
        except Exception as error:
            if not do_run: raise
            ctx.logger.error(f"UVM ip/{ip} failed: {error}")
            failures.append({"target": f"ip/{ip}", "error": str(error)})
    return {"chip": chip, "results": results, "failures": failures}
