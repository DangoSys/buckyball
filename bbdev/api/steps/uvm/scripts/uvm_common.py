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
from .waive import apply_waivers

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


def vcs_defines(domain, mapping, core, tile):
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
    if mapping.ball_dir == "smatmul":
        defs.append(f"+define+SMATMUL_TILE_ROWS={mapping.ball_params['tileRows']}")
    return defs


def smatmul_accumulator_filename(rtl_dir: Path) -> str:
    text = (rtl_dir / "SMatMulUnit.sv").read_text()
    modules = re.findall(r"\b(accumulator_\d+x\d+)\s+accumulator_ext\s*\(", text)
    if len(modules) != 1:
        raise ValueError(
            f"expected one SMatMul accumulator instance, found {modules!r}"
        )
    return f"{modules[0]}.sv"


def _filelist(
    verify_dir: Path,
    ball_dir: str,
    uvm_rel: str,
    rtl_rel: str,
    rtl_dir: Path,
    sim_dir: Path,
) -> str:
    src = verify_dir / "filelists" / f"{ball_dir}_ball.f"
    dst = sim_dir / f"{ball_dir}_ball.f"
    sim_dir.mkdir(parents=True, exist_ok=True)
    text = src.read_text()
    if "@SMATMUL_ACCUMULATOR@" in text:
        text = text.replace(
            "@SMATMUL_ACCUMULATOR@", smatmul_accumulator_filename(rtl_dir)
        )
    if "@IM2COL_BUFFER@" in text:
        modules = set(
            re.findall(
                r"\b(buf_\d+x\d+)\s+buf(?:Lo|Hi)_ext\s*\(",
                (rtl_dir / "LineBufferManager.sv").read_text(),
            )
        )
        if len(modules) != 1:
            raise ValueError(f"expected one Im2col buffer module, found {modules!r}")
        text = text.replace("@IM2COL_BUFFER@", f"{modules.pop()}.sv")
    dst.write_text(text.replace("@UVM@", uvm_rel).replace("@RTL@", rtl_rel))
    return str(dst.relative_to(verify_dir))


def load_ip(bbdir: str, name: str) -> dict:
    registry = Path(bbdir) / "verify" / "uvm" / "ip.toml"
    if name not in tomllib.loads(registry.read_text()).get("ips", []):
        raise ValueError(f"IP {name!r} is not registered in {registry}")
    root = (
        Path(bbdir)
        / "arch"
        / "src"
        / "main"
        / "scala"
        / "framework"
        / "mem-core"
        / name
    )
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
    if not targets:
        raise ValueError(f"registered IP {name!r} has no filelists under {resources}")
    return {
        "root": str(root.relative_to(bbdir)),
        "mill_module": name,
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


def check_uvm_result(result, target: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"UVM failed for {target}")
    counts = dict(
        re.findall(r"UVM_(ERROR|FATAL)\s*:\s*(\d+)", result.stdout + result.stderr)
    )
    if set(counts) != {"ERROR", "FATAL"}:
        raise RuntimeError(f"UVM report summary missing for {target}")
    if counts != {"ERROR": "0", "FATAL": "0"}:
        raise RuntimeError(
            f"UVM failed for {target}: errors={counts['ERROR']} fatals={counts['FATAL']}"
        )


def build_ip(bbdir: str, chip: str, name: str, ctx) -> dict:
    config = load_ip(bbdir, name)
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

    emit = (
        f"{shlex.quote(str(Path(bbdir) / 'result' / 'bin' / 'mill'))} "
        f"{shlex.quote(config['mill_module'] + '.run')}"
    )
    result = stream_run_logger(
        cmd=emit,
        logger=ctx.logger,
        cwd=str(root),
        stdout_prefix="uvm rtl",
        stderr_prefix="uvm rtl",
    )
    if result.returncode != 0:
        raise RuntimeError(f"RTL generation failed for IP {name}")
    apply_waivers(rtl)

    for target in config["targets"]:
        sim_dir = sim_root / target["name"]
        filelist = _ip_filelist(resources, target["filelist"], sim_dir, bbdir, rtl)
        simv = sim_dir / "simv"
        csrc = sim_dir / "csrc"
        hier = sim_dir / "cm_hier.cfg"
        hier.write_text(
            f"+tree {target['top']}.dut\n"
            f"+tree {target['top']}.source_if\n"
            f"+tree {target['top']}.sink_if\n"
        )
        script = (
            f"cd {shlex.quote(str(sim_dir))} && "
            f"rm -rf {shlex.quote(str(csrc))} {shlex.quote(str(simv))} {shlex.quote(str(simv))}.daidir && "
            f"mkdir -p {shlex.quote(str(csrc))} && "
            "vcs -full64 -sverilog -timescale=1ns/1ps -debug_access+all "
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


def run_ip(bbdir: str, chip: str, name: str, ctx, cov_root: str) -> dict:
    config = build_ip(bbdir, chip, name, ctx)
    root = Path(bbdir) / config["root"]
    manifest = root / config["model"]
    crate = tomllib.loads(manifest.read_text())["package"]["name"].replace("-", "_")
    model = manifest.parent / "target" / "debug" / f"lib{crate}"
    sim_root = root / "build" / "uvm" / chip

    coverage = []
    for target in config["targets"]:
        simv = sim_root / target["name"] / "simv"
        simv_q = shlex.quote(str(simv))
        cov_dir = Path(cov_root) / target["name"] / "coverage"
        script = (
            f'loader="$(patchelf --print-interpreter {simv_q})"; '
            f'library_path="$(patchelf --print-rpath {simv_q})"; '
            f'"$loader" --library-path "$library_path" {simv_q} -no_save '
            f"-sv_lib {shlex.quote(str(model))} +UVM_TESTNAME={shlex.quote(target['test'])} "
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
        check_uvm_result(result, f"IP {name} target={target['name']}")

        cov_dir.mkdir(parents=True, exist_ok=True)
        urg = (
            f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command "
            f"urg -dir {shlex.quote(str(simv))}.vdb -format text -report {shlex.quote(str(cov_dir))}"
        )
        result = stream_run_logger(
            cmd=urg,
            logger=ctx.logger,
            cwd=str(root),
            stdout_prefix="uvm urg",
            stderr_prefix="uvm urg",
        )
        if result.returncode != 0:
            raise RuntimeError(f"URG failed for IP {name} target={target['name']}")
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
    index = index_dir / "index.txt"
    index.write_text("\n".join(rows) + "\n")
    return {
        "chip": chip,
        "ip": name,
        "targets": [t["name"] for t in config["targets"]],
        "index": str(index),
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
    rtl_rel = os.path.relpath(rtl_dir, verify_dir)
    flist = _filelist(verify_dir, ball, uvm_rel, rtl_rel, rtl_dir, sim_dir)
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
        "vcs -full64 -sverilog -timescale=1ns/1ps -debug_access+all "
        "${=VCS_UVM_ARGS} "
        + " ".join(shlex.quote(d) for d in vcs_defines(domain, mapping, core, tile))
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
    bbdir: str, chip: str, ball: str | None, ip: str | None, ctx, do_run: bool
) -> dict:
    if ball is not None:
        return run_chip(bbdir, chip, ball, ctx, do_run)
    if ip is not None:
        if do_run:
            stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            run_root = log_dir(bbdir, chip, "verilog", stamp, "uvm", ip)
            return run_ip(bbdir, chip, ip, ctx, run_root)
        config = build_ip(bbdir, chip, ip, ctx)
        return {
            "chip": chip,
            "ip": ip,
            "targets": [target["name"] for target in config["targets"]],
        }

    targets = load_chip_uvm(bbdir, chip)
    results = [
        run_chip(bbdir, chip, target, ctx, do_run) for target in targets["balls"]
    ]
    for target in targets["ips"]:
        if do_run:
            stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            run_root = log_dir(bbdir, chip, "verilog", stamp, "uvm", target)
            results.append(run_ip(bbdir, chip, target, ctx, run_root))
        else:
            config = build_ip(bbdir, chip, target, ctx)
            results.append(
                {
                    "chip": chip,
                    "ip": target,
                    "targets": [item["name"] for item in config["targets"]],
                }
            )
    return {"chip": chip, "results": results}
