import fcntl
import os
import re
import shlex
import shutil
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from utils.path import design_output_root, load_design_config, log_dir, rtl_dir
from steps.mill.scripts.mill import elaborate_cmd
from utils.stream_run import stream_run_logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config/scripts"))


def load_ball(bbdir: str, design: str, selection: str):
    """The host and the one Ball of the design.pb of a Ball alone."""
    msg = load_design_config(bbdir, design, selection=selection)
    if not msg.HasField("ball"):
        raise ValueError(f"design.pb of {design} {selection} is not a Ball alone")
    balls = msg.ball.balldomain.balls
    if len(balls) != 1:
        raise ValueError(f"design.pb of {design} {selection} holds {len(balls)} Balls")
    return msg.ball.host, balls[0]


def vcs_defines(host, mapping, parameters):
    defs = [
        f"+define+BB_IN_BW={mapping.in_bw}",
        f"+define+BB_OUT_BW={mapping.out_bw}",
        f"+define+BB_MMIO_READ_BW={mapping.mmio_read_bw}",
        f"+define+BB_MMIO_WRITE_BW={mapping.mmio_write_bw}",
        f"+define+BB_BANK_ID_W={host.bank_id_bits}",
        f"+define+BB_GROUP_COUNT_W={host.bank_num.bit_length()}",
        f"+define+BB_GROUP_ID_W={(host.bank_num - 1).bit_length()}",
        f"+define+BB_BANK_ADDR_W={(host.bank_entries - 1).bit_length()}",
        f"+define+BB_ROB_ID_W={(host.rob_entries - 1).bit_length()}",
    ]
    for e in mapping.instructions:
        defs.append(f"+define+{e.mnemonic}_FUNCT7={e.funct7}")
    from ball_parameters import unpack_parameters

    typed = unpack_parameters(mapping)
    for define, parameter in parameters.items():
        defs.append(f"+define+{define}={getattr(typed, parameter)}")
    return defs


def _filelist(
    verify_dir: Path, ball_dir: str, uvm_dir: Path, rtl_dir: Path, sim_dir: Path
) -> str:
    """The Ball's filelist with every path absolute, so that tools run in
    `sim_dir` and leave nothing in the Ball's source directory."""
    src = verify_dir / "filelists" / f"{ball_dir}_ball.f"
    dst = sim_dir / f"{ball_dir}_ball.f"
    sim_dir.mkdir(parents=True, exist_ok=True)
    text = (
        src.read_text()
        .replace("@UVM@", str(uvm_dir))
        .replace("@RTL@", str(rtl_dir.resolve()))
    )
    lines = []
    for line in text.splitlines():
        entry = line.strip()
        if entry.startswith("+incdir+"):
            line = "+incdir+" + str(verify_dir / entry.removeprefix("+incdir+"))
        elif entry and not entry.startswith(("+", "-", "//")):
            line = str(verify_dir / entry)
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n")
    return str(dst)


def check_uvm_result(result, target: str) -> None:
    output = result.stdout + result.stderr
    # An RTL assertion that fails without ending the run (a third-party
    # `[ASSERT FAILED]` print, a concurrent SVA's `Offending`) fails the test.
    failed = re.findall(r"^.*(?:\[ASSERT FAILED\]|Offending ').*$", output, re.M)
    if failed:
        raise RuntimeError(
            f"RTL assertion failed in {target} ({len(failed)}): {failed[0].strip()}"
        )
    if result.returncode != 0:
        raise RuntimeError(f"UVM failed for {target}")
    counts = dict(re.findall(r"UVM_(ERROR|FATAL)\s*:\s*(\d+)", output))
    if set(counts) != {"ERROR", "FATAL"}:
        raise RuntimeError(f"UVM report summary missing for {target}")
    if counts != {"ERROR": "0", "FATAL": "0"}:
        raise RuntimeError(
            f"UVM failed for {target}: errors={counts['ERROR']} fatals={counts['FATAL']}"
        )


def check_illegal_result(result, target: str, message: str) -> None:
    """An illegal-input test passes when the RTL rejects the command: exactly
    the assertion `message` fails and ends the run, nothing else fails."""
    output = result.stdout + result.stderr
    failed = re.findall(r"Assertion failed: ([^\r\n]+)", output)
    if failed != [message]:
        raise RuntimeError(
            f"{target}: expected the assertion {message!r}, got {failed}"
        )
    if "$finish called" not in output:
        raise RuntimeError(f"{target}: the assertion did not end the run")
    if re.search(r"\[ASSERT FAILED\]|Offending '|UVM_FATAL @", output):
        raise RuntimeError(f"{target}: another check failed besides {message!r}")


def build_ball(bbdir: str, design: str, selection: str, ctx) -> Path:
    """Elaborate the BallDomain of the Ball alone into arch/uvm, compile its
    case generator and its testbench; return the simulation directory."""
    host, mapping = load_ball(bbdir, design, selection)
    ball = mapping.component
    verify_dir = Path(bbdir) / mapping.source_path / "verify"
    casegen = verify_dir / "casegen" / "Cargo.toml"
    rtl = Path(rtl_dir(bbdir, design, "uvm", selection=selection))
    if rtl.exists():
        shutil.rmtree(rtl)
    rtl.mkdir(parents=True)
    r = stream_run_logger(
        cmd=elaborate_cmd(design, selection, "verilog", str(rtl)),
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="uvm verilog",
        stderr_prefix="uvm verilog",
    )
    if r.returncode != 0:
        raise RuntimeError(f"elaboration failed for {ball}")
    sim_dir = Path(design_output_root(bbdir, design, selection=selection)) / "uvm"
    flist = _filelist(verify_dir, ball, Path(bbdir) / "verify" / "uvm", rtl, sim_dir)
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
    simv = sim_dir / "simv"
    csrc = sim_dir / "csrc"
    hier = sim_dir / "cm_hier.cfg"
    hier.write_text("+tree tb_top.dut\n")
    script = (
        f"cd {shlex.quote(str(sim_dir))} && "
        f"rm -rf {shlex.quote(str(csrc))} {shlex.quote(str(simv))} "
        f"{shlex.quote(str(simv))}.daidir {shlex.quote(str(simv))}.vdb && "
        f"mkdir -p {shlex.quote(str(csrc))} && "
        "vcs -full64 -sverilog -timescale=1ns/1ps -debug_access+all -top tb_top "
        "${=VCS_UVM_ARGS} "
        + " ".join(shlex.quote(d) for d in vcs_defines(host, mapping, parameters))
        + f" -cm line+cond+tgl+assert -cm_libs yv -cm_hier {shlex.quote(str(hier))} "
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
    return sim_dir


def ball_sim(bbdir: str, design: str, selection: str, ctx):
    """Build the simulation of a Ball alone; return its mapping, its verify/
    directory, its simulation directory and `run(test, prefix, extra)`, which
    runs one UVM test on it and returns the result."""
    sim_dir = build_ball(bbdir, design, selection, ctx)
    host, mapping = load_ball(bbdir, design, selection)
    verify_dir = Path(bbdir) / mapping.source_path / "verify"
    simv_q = shlex.quote(str(sim_dir / "simv"))
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
    verify_config = sim_dir / "verify_config.env"
    verify_config.write_text(
        f"bank_entries={host.bank_entries}\nball_id={mapping.id}\n"
    )

    def run(test: str, prefix: str, extra: str):
        script = (
            f"cd {shlex.quote(str(sim_dir))} && "
            f'loader="$(patchelf --print-interpreter {simv_q})"; '
            f'library_path="$(patchelf --print-rpath {simv_q})"; '
            f"BB_VERIFY_CONFIG={shlex.quote(str(verify_config))} "
            f'"$loader" --library-path "$library_path" {simv_q} -no_save '
            f"-sv_lib {shlex.quote(str(dpi))} "
            f"+UVM_TESTNAME={shlex.quote(test)} +BID={mapping.id} "
            f"-cm line+cond+tgl+assert -cm_name {shlex.quote(test)}{extra}"
        )
        cmd = f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command zsh -ic {shlex.quote(script)}"
        return stream_run_logger(
            cmd=cmd,
            logger=ctx.logger,
            cwd=bbdir,
            stdout_prefix=prefix,
            stderr_prefix=prefix,
        )

    return mapping, verify_dir, sim_dir, run


def run_ball(bbdir: str, design: str, selection: str, ctx, run_root: str) -> Path:
    """Run the functional test and the illegal-input tests; write coverage/
    and the switching activity of the functional test, activity.saif, under
    `run_root`; return the SAIF. Every instruction of the functional test is
    checked against the Ball's BEMU cycle model by bb_blink_cycles."""
    mapping, verify_dir, sim_dir, run = ball_sim(bbdir, design, selection, ctx)
    ball = mapping.component
    activity = Path(run_root) / "activity.saif"
    activity.parent.mkdir(parents=True, exist_ok=True)
    # UVM's $finish would end the run before the report: hold it with
    # endofsim noexit (needs -ucli2Proc), then write the SAIF of the Ball.
    power_tcl = sim_dir / "activity.tcl"
    power_tcl.write_text(
        "config endofsim noexit\n"
        # rtl_on also records reg variables (the registers PrimeTime matches
        # by name); the default records nets only.
        "power -gate_level rtl_on\n"
        "power tb_top.dut\n"
        "power -enable\n"
        "run\n"
        "power -disable\n"
        f"power -report {activity} 1e-09 tb_top.dut\n"
        "quit\n"
    )
    test = f"{ball}_ball_test"
    r = run(test, "uvm run", f" -ucli -ucli2Proc -do {shlex.quote(str(power_tcl))}")
    check_uvm_result(r, f"{ball} test={test}")
    if not activity.is_file():
        raise RuntimeError(f"simv wrote no {activity}")
    # Illegal-input tests: each must end on exactly the RTL assertion that
    # verify/illegal.toml names for it.
    illegal = verify_dir / "illegal.toml"
    for test, message in (
        tomllib.loads(illegal.read_text()).items() if illegal.is_file() else ()
    ):
        r = run(test, "uvm illegal", "")
        check_illegal_result(r, f"{ball} test={test}", message)
    cov_dir = Path(run_root) / "coverage"
    cov_dir.mkdir(parents=True, exist_ok=True)
    urg = (
        f"nix develop {shlex.quote(str(Path(bbdir) / 'verify'))} --command "
        # urg1 races its two threads and crashes at random while writing the
        # toggle report; on one CPU the threads never run at once.
        f"taskset -c {min(os.sched_getaffinity(0))} urg -full64 -dir {shlex.quote(str(sim_dir / 'simv'))}.vdb "
        f"-format text -report {shlex.quote(str(cov_dir))}"
    )
    r = stream_run_logger(
        cmd=urg,
        logger=ctx.logger,
        cwd=str(sim_dir),
        stdout_prefix="uvm urg",
        stderr_prefix="uvm urg",
    )
    if r.returncode != 0:
        raise RuntimeError(f"urg failed for {ball}")
    return activity


def run_performance(bbdir: str, design: str, selection: str, ctx) -> dict:
    """Build a Ball alone and run its throughput and latency tests: one
    instruction each against the ideal bank model, at the workloads that set
    each timing path's rate and its shortest latency. A test fails unless the
    cycles from its command handshake to its response equal those of the
    Ball's BEMU cycle model (emu/src/latency.rs); return the cycles they print."""
    _, mapping = load_ball(bbdir, design, selection)
    lock = Path(design_output_root(bbdir, design, selection=selection)) / "uvm.lock"
    with open(lock, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        _, _, _, run = ball_sim(bbdir, design, selection, ctx)
        performance = {}
        for kind in ("throughput", "latency"):
            test = f"{mapping.component}_{kind}_test"
            r = run(test, f"uvm {kind}", "")
            check_uvm_result(r, f"{mapping.component} test={test}")
            measured = {
                (instruction, int(rows)): int(cycles)
                for instruction, rows, cycles in re.findall(
                    r"BBPERF (\S+) (\d+) (\d+)", r.stdout
                )
            }
            performance[kind] = [
                {"instruction": instruction, "iter": rows, "cycles": cycles}
                for (instruction, rows), cycles in sorted(measured.items())
            ]
    return {"design": design, "ball": mapping.component, "performance": performance}


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


def run_design(bbdir: str, design: str, selection: str, ctx, do_run: bool) -> dict:
    """Build, and with `do_run` run, the UVM test of a Ball alone. Runs of one
    Ball share its RTL and simulation directories, so they take turns."""
    _, mapping = load_ball(bbdir, design, selection)
    ball = mapping.component
    lock = Path(design_output_root(bbdir, design, selection=selection)) / "uvm.lock"
    with open(lock, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not do_run:
            build_ball(bbdir, design, selection, ctx)
            return {"design": design, "ball": ball}
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_root = log_dir(
            bbdir, design, "verilog", stamp, "uvm", ball, selection=selection
        )
        cov = os.path.join(run_root, "coverage")
        activity = run_ball(bbdir, design, selection, ctx, run_root)
    if not (Path(cov) / "dashboard.txt").is_file():
        raise RuntimeError(f"urg wrote no dashboard.txt under {cov}")
    return {
        "design": design,
        "ball": ball,
        "log": run_root,
        "coverage": dashboard_summary(cov),
        "activity": str(activity),
    }
