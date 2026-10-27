import os
import re
import shlex
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from utils.path import design_arch_root, log_dir
from steps.mill.scripts.mill import mill_cmd
from utils.stream_run import stream_run_logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config/scripts"))


def load_design(bbdir: str, design: str, selection: str):
    from utils.path import load_design_config

    msg = load_design_config(bbdir, design, selection=selection)
    if not msg.name or not msg.device.cores:
        raise ValueError(f"empty design.pb of {design} {selection}")
    return msg


def ball_domain(design, ball: str | None = None):
    domains = [
        core.accelerator.balldomain
        for core in design.device.cores
        if core.HasField("accelerator")
    ]
    domains.extend(thread.balldomain for thread in design.threads)
    domains = [domain for domain in domains if domain.balls]
    if not domains:
        raise ValueError("design.pb has no BallDomain instances")
    if ball is None:
        domain = domains[0]
        if any(other.balls != domain.balls for other in domains[1:]):
            raise ValueError(
                "design.pb has different BallDomains; select a Ball explicitly"
            )
        return domain
    matches = [
        (domain, mapping)
        for domain in domains
        for mapping in domain.balls
        if mapping.component == ball
    ]
    if not matches:
        raise ValueError(f"ball {ball!r} not in design.pb")
    domain, mapping = matches[0]
    if any(other != mapping for _, other in matches[1:]):
        raise ValueError(f"ball {ball!r} has different instance definitions")
    return domain


def selected_mappings(domain, ball: str | None):
    if ball is None:
        return list(domain.balls)
    for m in domain.balls:
        if m.component == ball:
            return [m]
    raise ValueError(f"ball {ball!r} not in design.pb")


def ball_owner(design, ball):
    owners = []
    for core in design.device.cores:
        accelerators = [core.accelerator] if core.HasField("accelerator") else []
        accelerators.extend(design.threads[index] for index in core.thread_indices)
        owners.extend(
            (acc, core)
            for acc in accelerators
            if any(item.component == ball for item in acc.balldomain.balls)
        )
    if not owners:
        raise ValueError(f"ball {ball!r} not in design.pb")
    # The first owner elaborates the unsuffixed Ball module that tb_top
    # instantiates. Owners may differ only in bank count, which changes the
    # group and column port widths of their own module copies.
    accelerator, core = owners[0]
    for other, owner in owners[1:]:
        if (
            other.mem.bank.entries != accelerator.mem.bank.entries
            or other.mem.bank.width != accelerator.mem.bank.width
            or other.mem.bank.mask_len != accelerator.mem.bank.mask_len
            or other.frontend.rob_entries != accelerator.frontend.rob_entries
            or other.frontend.sub_rob_depth != accelerator.frontend.sub_rob_depth
            or owner.shared_mem != core.shared_mem
            or owner.virtual_bank_count != core.virtual_bank_count
        ):
            raise ValueError(
                f"ball {ball!r} has different verification interfaces across instances"
            )
    return accelerator, core


def vcs_defines(domain, mapping, thread, core, parameters):
    shared_bank_num = (
        core.shared_mem.entries // core.shared_mem.bank_entries
        if core.shared_mem.enable
        else 0
    )
    max_group_count = max(thread.mem.bank.num, shared_bank_num)
    defs = [
        f"+define+BB_IN_BW={mapping.in_bw}",
        f"+define+BB_OUT_BW={mapping.out_bw}",
        f"+define+BB_MMIO_READ_BW={mapping.mmio_read_bw}",
        f"+define+BB_MMIO_WRITE_BW={mapping.mmio_write_bw}",
        f"+define+BB_BANK_ID_W={thread.frontend.bank_id_len}",
        f"+define+BB_GROUP_COUNT_W={max_group_count.bit_length()}",
        f"+define+BB_GROUP_ID_W={(max_group_count - 1).bit_length()}",
        f"+define+BB_BANK_ADDR_W={(thread.mem.bank.entries - 1).bit_length()}",
        f"+define+BB_ROB_ID_W={(thread.frontend.rob_entries - 1).bit_length()}",
    ]
    for e in mapping.instructions:
        defs.append(f"+define+{e.mnemonic}_FUNCT7={e.funct7}")
    from ball_parameters import unpack_parameters

    typed = unpack_parameters(mapping)
    for define, parameter in parameters.items():
        defs.append(f"+define+{define}={getattr(typed, parameter)}")
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
    dst.write_text(
        text.replace("@UVM@", uvm_rel).replace("@RTL@", str(rtl_dir.resolve()))
    )
    return str(dst)


def check_uvm_result(
    result, target: str, expected_assertion: str | None = None
) -> None:
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
            raise RuntimeError(
                f"Expected RTL assertion did not terminate {target}: {expected_assertion}"
            )
        return
    if result.returncode != 0:
        raise RuntimeError(f"UVM failed for {target}")
    counts = dict(re.findall(r"UVM_(ERROR|FATAL)\s*:\s*(\d+)", output))
    if set(counts) != {"ERROR", "FATAL"}:
        raise RuntimeError(f"UVM report summary missing for {target}")
    if counts != {"ERROR": "0", "FATAL": "0"}:
        raise RuntimeError(
            f"UVM failed for {target}: errors={counts['ERROR']} fatals={counts['FATAL']}"
        )


def build_ball(
    bbdir: str, design_name: str, selection: str, mill_cfg: str, domain, mapping, ctx
) -> None:
    ball = mapping.component
    verify_dir = Path(bbdir) / "examples" / "prefab" / "balls" / ball / "verify"
    casegen = verify_dir / "casegen" / "Cargo.toml"
    rtl_dir = Path(design_arch_root(bbdir, design_name, selection=selection)) / (
        f"verilog/{mill_cfg}"
    )
    design = load_design(bbdir, design_name, selection)
    thread, core = ball_owner(design, ball)
    sim_dir = verify_dir.parent / "output" / "uvm" / design_name / selection
    uvm_rel = os.path.relpath(Path(bbdir) / "verify" / "uvm", verify_dir)
    flist = _filelist(verify_dir, ball, uvm_rel, rtl_dir, sim_dir)
    parameters_path = verify_dir / "parameters.toml"
    parameters = (
        tomllib.loads(parameters_path.read_text())["defines"]
        if parameters_path.is_file()
        else {}
    )
    cargo = (
        f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command "
        f"cargo build --manifest-path {shlex.quote(str(casegen))} "
        f"--target-dir {shlex.quote(str(verify_dir.parent / 'output' / 'casegen'))}"
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
        + " ".join(
            shlex.quote(d)
            for d in vcs_defines(domain, mapping, thread, core, parameters)
        )
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
    bbdir: str,
    design_name: str,
    selection: str,
    mill_cfg: str,
    domain,
    mapping,
    ctx,
    cov_dir: str,
) -> None:
    ball = mapping.component
    verify_dir = Path(bbdir) / "examples" / "prefab" / "balls" / ball / "verify"
    simv = verify_dir.parent / "output" / "uvm" / design_name / selection / "simv"
    build_ball(bbdir, design_name, selection, mill_cfg, domain, mapping, ctx)
    crate = tomllib.loads((verify_dir / "casegen" / "Cargo.toml").read_text())[
        "package"
    ]["name"]
    dpi = (
        verify_dir.parent
        / "output"
        / "casegen"
        / "debug"
        / f"lib{crate.replace('-', '_')}"
    )
    test = f"{ball}_ball_test"
    verify_config = (
        verify_dir.parent
        / "output"
        / "uvm"
        / design_name
        / selection
        / "verify_config.env"
    )
    design = load_design(bbdir, design_name, selection)
    thread, _ = ball_owner(design, ball)
    verify_config.write_text(
        f"bank_entries={thread.mem.bank.entries}\nball_id={mapping.id}\n"
    )
    simv_q = shlex.quote(str(simv))
    script = (
        f"cd {shlex.quote(str(verify_dir))} && "
        f'loader="$(patchelf --print-interpreter {simv_q})"; '
        f'library_path="$(patchelf --print-rpath {simv_q})"; '
        f"BB_VERIFY_CONFIG={shlex.quote(str(verify_config))} "
        f'"$loader" --library-path "$library_path" {simv_q} -no_save '
        f"-sv_lib {shlex.quote(str(dpi))} "
        f"+UVM_TESTNAME={shlex.quote(test)} +BID={mapping.id} "
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


def run_design(
    bbdir: str, design: str, selection: str, ball: str | None, ctx, do_run: bool
) -> dict:
    msg = load_design(bbdir, design, selection)
    domain = ball_domain(msg, ball)
    mill_cfg = msg.simulation.top
    if not mill_cfg:
        raise ValueError(f"design {design!r} has no sims.verilator for UVM")
    maps = selected_mappings(domain, ball)
    ran = []
    covs = []
    run_root = None
    if do_run:
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        run_name = "all" if ball is None else maps[0].component
        run_root = log_dir(
            bbdir, design, "verilog", stamp, "uvm", run_name, selection=selection
        )
    for m in maps:
        if do_run:
            cov = (
                os.path.join(run_root, m.component, "coverage")
                if ball is None
                else os.path.join(run_root, "coverage")
            )
            run_ball(bbdir, design, selection, mill_cfg, domain, m, ctx, cov)
            if not (Path(cov) / "dashboard.txt").is_file():
                raise RuntimeError(f"urg wrote no dashboard.txt under {cov}")
            covs.append((m.component, cov))
        else:
            build_ball(bbdir, design, selection, mill_cfg, domain, m, ctx)
        ran.append(m.component)
    info = {"design": design, "mill": mill_cfg, "balls": ran}
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
