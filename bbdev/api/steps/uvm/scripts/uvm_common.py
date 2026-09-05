import os
import shlex
import sys
import tomllib
from datetime import datetime
from pathlib import Path

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
config_scripts = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "scripts"))
if config_scripts not in sys.path:
    sys.path.insert(0, config_scripts)

import chip_pb2
from utils.path import get_buckyball_path, log_dir
from utils.stream_run import stream_run_logger


def load_chip(bbdir: str, chip: str):
    path = Path(bbdir) / "examples" / "chips" / chip / "configs" / "generated" / "chip.pb"
    msg = chip_pb2.Chip()
    msg.ParseFromString(path.read_bytes())
    if not msg.name or not msg.cores:
        raise ValueError(f"empty chip.pb: {path}")
    return msg


def ball_domain(chip):
    d0 = chip.cores[0].balldomain
    key = [(m.ball_id, m.ball_dir, m.in_bw, m.out_bw) for m in d0.mappings]
    isa = [(e.mnemonic, e.funct7, e.bid) for e in d0.isa]
    for core in chip.cores[1:]:
        k = [(m.ball_id, m.ball_dir, m.in_bw, m.out_bw) for m in core.balldomain.mappings]
        i = [(e.mnemonic, e.funct7, e.bid) for e in core.balldomain.isa]
        if k != key or i != isa:
            raise ValueError("chip.pb cores have different balldomains")
    return d0


def selected_mappings(domain, ball: str | None):
    if ball is None:
        return list(domain.mappings)
    for m in domain.mappings:
        if m.ball_dir == ball:
            return [m]
    raise ValueError(f"ball {ball!r} not in chip.pb")


def vcs_defines(domain, mapping):
    defs = [
        f"+define+BB_IN_BW={mapping.in_bw}",
        f"+define+BB_OUT_BW={mapping.out_bw}",
        f"+define+BB_MMIO_READ_BW={mapping.mmio_read_bw}",
        f"+define+BB_MMIO_WRITE_BW={mapping.mmio_write_bw}",
    ]
    for e in domain.isa:
        if e.bid == mapping.ball_id:
            defs.append(f"+define+{e.mnemonic}_FUNCT7={e.funct7}")
    return defs


def _filelist(verify_dir: Path, ball_dir: str, uvm_rel: str, rtl_rel: str, sim_dir: Path) -> str:
    src = verify_dir / "filelists" / f"{ball_dir}_ball.f"
    dst = sim_dir / f"{ball_dir}_ball.f"
    sim_dir.mkdir(parents=True, exist_ok=True)
    text = src.read_text()
    dst.write_text(text.replace("@UVM@", uvm_rel).replace("@RTL@", rtl_rel))
    return str(dst.relative_to(verify_dir))


def build_ball(bbdir: str, chip_name: str, mill_cfg: str, domain, mapping, ctx) -> None:
    ball = mapping.ball_dir
    verify_dir = Path(bbdir) / "examples" / "balls" / ball / "verify"
    casegen = verify_dir / "casegen" / "Cargo.toml"
    rtl_dir = Path(bbdir) / "arch" / "build" / chip_name / mill_cfg
    sim_dir = verify_dir / "build" / chip_name
    uvm_rel = os.path.relpath(Path(bbdir) / "verify" / "uvm", verify_dir)
    rtl_rel = os.path.relpath(rtl_dir, verify_dir)
    flist = _filelist(verify_dir, ball, uvm_rel, rtl_rel, sim_dir)
    cargo = (
        f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command "
        f"cargo build --manifest-path {shlex.quote(str(casegen))}"
    )
    r = stream_run_logger(cmd=cargo, logger=ctx.logger, cwd=bbdir, stdout_prefix="uvm dpi", stderr_prefix="uvm dpi")
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
        + " ".join(shlex.quote(d) for d in vcs_defines(domain, mapping))
        + f" -cm line+cond+tgl+assert -cm_hier {shlex.quote(str(hier))} "
        f"-Mdir={shlex.quote(str(csrc))} -o {shlex.quote(str(simv))} "
        f"-f {shlex.quote(flist)}"
    )
    vcs = f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command zsh -c {shlex.quote(script)}"
    r = stream_run_logger(cmd=vcs, logger=ctx.logger, cwd=bbdir, stdout_prefix="uvm vcs", stderr_prefix="uvm vcs")
    if r.returncode != 0:
        raise RuntimeError(f"vcs failed for {ball}")


def run_ball(bbdir: str, chip_name: str, mill_cfg: str, domain, mapping, ctx, cov_dir: str) -> None:
    ball = mapping.ball_dir
    verify_dir = Path(bbdir) / "examples" / "balls" / ball / "verify"
    simv = verify_dir / "build" / chip_name / "simv"
    if not simv.is_file():
        build_ball(bbdir, chip_name, mill_cfg, domain, mapping, ctx)
    crate = tomllib.loads((verify_dir / "casegen" / "Cargo.toml").read_text())["package"]["name"]
    dpi = verify_dir / "casegen" / "target" / "debug" / f"lib{crate.replace('-', '_')}"
    test = f"{ball}_ball_test"
    script = (
        f"cd {shlex.quote(str(verify_dir))} && "
        f"env LD_LIBRARY_PATH=\"$VCS_RUNTIME_LIBRARY_PATH\" {shlex.quote(str(simv))} "
        f"-sv_lib {shlex.quote(str(dpi))} "
        f"+UVM_TESTNAME={shlex.quote(test)} +BID={mapping.ball_id} "
        f"-cm line+cond+tgl+assert -cm_name {shlex.quote(test)}"
    )
    cmd = f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command zsh -ic {shlex.quote(script)}"
    r = stream_run_logger(cmd=cmd, logger=ctx.logger, cwd=bbdir, stdout_prefix="uvm run", stderr_prefix="uvm run")
    if r.returncode != 0:
        raise RuntimeError(f"uvm run failed for {ball} test={test}")
    Path(cov_dir).mkdir(parents=True, exist_ok=True)
    urg = (
        f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command "
        f"urg -dir {shlex.quote(str(simv))}.vdb -format text -report {shlex.quote(cov_dir)}"
    )
    r = stream_run_logger(cmd=urg, logger=ctx.logger, cwd=str(verify_dir), stdout_prefix="uvm urg", stderr_prefix="uvm urg")
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
    domain = ball_domain(msg)
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
                f"{s.get('ASSERT', '--')} {s['GROUP']} pass"
            )
        index = index_dir / "index.txt"
        index.write_text("\n".join(body) + "\n")
        info["index"] = str(index)
        info["log"] = run_root
    elif do_run:
        info["log"] = run_root
    return info
