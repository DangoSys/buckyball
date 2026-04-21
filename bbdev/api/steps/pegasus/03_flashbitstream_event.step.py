"""
pegasus flashbitstream event handler

Sequence:
  1. Detect XDMA PCIe BDF
  2. Flash via Vivado + program_fpga.tcl  (JTAG — independent of PCIe)
  3. PCIe remove + rescan so new bitstream's PCIe IP re-enumerates
"""
import os
import sys
import time
import socket
import subprocess
import glob
import re

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "pegasus-flashbitstream",
    "description": "Flash AU280 bitstream via Vivado hw_server + PCIe remove/rescan",
    "flows": ["pegasus"],
    "triggers": [queue("pegasus.flashbitstream")],
    "enqueues": [],
}

PCI_DEVICES = "/sys/bus/pci/devices"
XILINX_VENDOR = "0x10ee"
XDMA_DEVICE_IDS = {"0x903f"}


def _find_xdma_bdf() -> str | None:
    """Return the extended BDF (e.g. '0000:65:00.0') bound to xdma driver."""
    for entry in os.listdir(PCI_DEVICES):
        driver_link = os.path.join(PCI_DEVICES, entry, "driver")
        if not os.path.islink(driver_link):
            continue
        driver_name = os.path.basename(os.readlink(driver_link))
        if driver_name == "xdma":
            return entry   # e.g. '0000:65:00.0'
    return None


def _find_xdma_pci_bdf() -> str | None:
    """Return first XDMA-like Xilinx PCIe endpoint even if driver is not bound."""
    for entry in os.listdir(PCI_DEVICES):
        vendor_file = os.path.join(PCI_DEVICES, entry, "vendor")
        device_file = os.path.join(PCI_DEVICES, entry, "device")
        if not (os.path.exists(vendor_file) and os.path.exists(device_file)):
            continue
        vendor = open(vendor_file, "r", encoding="utf-8").read().strip().lower()
        device = open(device_file, "r", encoding="utf-8").read().strip().lower()
        if vendor == XILINX_VENDOR and device in XDMA_DEVICE_IDS:
            return entry
    return None


def _bridge_bdf(extended_bdf: str) -> str | None:
    """Return the bridge BDF for a given extended device BDF."""
    dev_path = os.path.realpath(os.path.join(PCI_DEVICES, extended_bdf))
    bridge_path = os.path.dirname(dev_path)
    bridge_bdf = os.path.basename(bridge_path)
    if bridge_bdf.startswith("0000:"):
        return bridge_bdf
    return None


def _ensure_hw_server(port: int = 3121, logger=None) -> bool:
    """Ensure hw_server is listening on the given port; start it if not."""
    try:
        with socket.create_connection(("localhost", port), timeout=2):
            return True
    except OSError:
        pass
    if logger:
        logger.info(f"[pegasus] hw_server not on port {port}, starting ...")
    subprocess.Popen(
        ["hw_server"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(15):
        time.sleep(1)
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                if logger:
                    logger.info(f"[pegasus] hw_server ready on port {port}")
                return True
        except OSError:
            pass
    if logger:
        logger.error(f"[pegasus] hw_server failed to start on port {port}")
    return False


def _setpci(bdf: str, reg: str, val: str) -> bool:
    r = subprocess.run(["sudo", "setpci", "-s", bdf, f"{reg}={val}"],
                       capture_output=True)
    return r.returncode == 0


def _sysfs_write(path: str, value: str, logger) -> bool:
    """Write to a sysfs file via 'sudo tee' (needed for root-owned sysfs entries)."""
    r = subprocess.run(
        f"echo {value} | sudo tee {path}",
        shell=True, capture_output=True,
    )
    if r.returncode != 0:
        logger.error(f" sysfs write failed: {path} <- {value}: {r.stderr.decode().strip()}")
    return r.returncode == 0


def _pcie_disconnect(extended_bdf: str, bridge: str | None, logger) -> None:
    """Clear SERR/fatal-error bits and remove device from PCIe bus."""
    if bridge:
        logger.info(f" clearing SERR bit on bridge {bridge}")
        _setpci(bridge, "COMMAND", "0000:0100")
        time.sleep(0.5)
        logger.info(f" clearing fatal-error bit on bridge {bridge}")
        _setpci(bridge, "CAP_EXP+8.w", "0000:0004")
        time.sleep(0.5)
    remove_path = os.path.join(PCI_DEVICES, extended_bdf, "remove")
    if os.path.exists(remove_path):
        logger.info(f" removing PCIe device {extended_bdf}")
        if _sysfs_write(remove_path, "1", logger):
            time.sleep(2)


def _pcie_reconnect(bridge: str | None, device_bdf: str | None, logger) -> None:
    """Rescan PCIe bus and re-enable memory-mapped transfers."""
    if bridge:
        rescan_path = os.path.join(PCI_DEVICES, bridge, "rescan")
        if os.path.exists(rescan_path):
            logger.info(f" rescanning bridge {bridge}")
            _sysfs_write(rescan_path, "1", logger)
            time.sleep(1)
    logger.info(" global PCIe rescan")
    _sysfs_write("/sys/bus/pci/rescan", "1", logger)
    time.sleep(2)
    # Re-detect after rescan: prefer xdma-bound BDF, fall back to raw PCIe endpoint.
    new_bdf = _find_xdma_bdf() or _find_xdma_pci_bdf()
    if not new_bdf:
        for _ in range(5):
            _sysfs_write("/sys/bus/pci/rescan", "1", logger)
            time.sleep(1)
            new_bdf = _find_xdma_bdf() or _find_xdma_pci_bdf()
            if new_bdf:
                break
    if not new_bdf:
        raise RuntimeError("xdma pci endpoint not found after PCIe rescan")
    logger.info(f" enabling MEM+BUSMSTR on {new_bdf}")
    _setpci(new_bdf, "COMMAND.W", "0006")


def _run_checked(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)


def _reload_xdma(logger) -> None:
    """Align XDMA lifecycle with FireSim: unload/load with poll_mode=1."""
    logger.info(" reloading xdma kernel module")
    loaded = subprocess.run(
        "lsmod | grep -wq xdma", shell=True
    ).returncode == 0
    if loaded:
        _run_checked("sudo rmmod xdma")
    _run_checked("sudo sh -c 'insmod $(find /lib/modules/$(uname -r) -name \"xdma.ko\") poll_mode=1'")
    _run_checked("sudo chmod a+rw /dev/xdma*")
    devs = glob.glob("/dev/xdma*_h2c_0")
    if not devs:
        raise RuntimeError("xdma driver loaded but /dev/xdma*_h2c_0 not found")
    logger.info(f" xdma ready: {devs[0]}")


def _parse_mig_state(output: str) -> str:
    text = output.lower()
    if "calibration is still in-progress" in text:
        return "busy"

    pairs = re.findall(r"^MIG_PROP\s+([^=]+)=(.*)$", output, flags=re.M)
    if not pairs:
        return "unknown"

    norm = [(k.strip().lower(), v.strip().lower()) for k, v in pairs]
    prop_map = {k: v for k, v in norm}

    init_done = prop_map.get("init_calib_complete")
    if init_done is not None:
        if init_done in {"1", "true", "yes"}:
            return "ready"
        if init_done in {"0", "false", "no"}:
            return "busy"

    if any(re.search(r"(in[-_ ]progress|busy)", v) for _, v in norm):
        return "busy"

    fail_status = prop_map.get("calibration_fail.status")
    has_pass = any(k.startswith("cal_status.") and v == "pass" for k, v in norm)
    if has_pass and fail_status in {"false", "0", "no"}:
        return "ready"

    status_ready = any(
        ("status" in k or "calib" in k) and re.search(r"(done|pass|complete|success)", v)
        for k, v in norm
    )
    if status_ready:
        return "ready"

    return "unknown"


def _wait_ddr_calibration(hw_server_url: str, serial: str, timeout_sec: int, tcl_script: str, logger) -> None:
    if timeout_sec <= 0:
        logger.info(" skip DDR calibration polling because timeout <= 0")
        return
    deadline = time.time() + timeout_sec
    poll_sec = 5
    logger.info(f" polling DDR calibration status for up to {timeout_sec}s")
    while True:
        cmd = (
            f"vivado -mode tcl -source {tcl_script}"
            f" -tclargs -hw_server {hw_server_url}"
        )
        if serial:
            cmd += f" -serial {serial}"
        result = stream_run_logger(
            cmd=cmd,
            logger=logger,
            stdout_prefix="pegasus mig",
            stderr_prefix="pegasus mig",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mig status query failed: rc={result.returncode} stdout={result.stdout} stderr={result.stderr}"
            )
        merged = f"{result.stdout}\n{result.stderr}"
        state = _parse_mig_state(merged)
        if state == "ready":
            logger.info(" DDR calibration reported complete")
            return
        if state == "unknown":
            raise RuntimeError("DDR calibration state unknown: no explicit MIG calibration status")
        now = time.time()
        if now >= deadline:
            raise RuntimeError("DDR calibration did not complete before timeout")
        remain = int(max(0, deadline - now))
        logger.info(f" DDR calibration still in progress, {remain}s remaining")
        time.sleep(min(poll_sec, max(1, deadline - now)))


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    # ── 0. ensure hw_server is running (Vivado can't launch it in bbdev env) ─
    hw_server_url = input_data.get("hw_server", "localhost:3121")
    hw_port = int(hw_server_url.split(":")[-1])
    if not _ensure_hw_server(hw_port, ctx.logger):
        await check_result(ctx, 1, continue_run=False,
                           extra_fields={"error": "hw_server_unavailable"}, trace_id=origin_tid)
        return

    bitstream = input_data.get(
        "bitstream",
        os.path.join(bbdir, "thirdparty", "pegasus", "vivado", "build", "PegasusTop.bit"),
    )
    serial = input_data.get("serial", "")   # empty = auto (first target)
    ddr_settle_sec = int(input_data.get("ddr_settle_sec", 180))
    tcl_script = os.path.join(
        bbdir, "thirdparty", "pegasus", "vivado", "scripts", "program_fpga.tcl"
    )
    mig_status_tcl = os.path.join(
        bbdir, "thirdparty", "pegasus", "vivado", "scripts", "query_mig_status.tcl"
    )

    if not os.path.exists(bitstream):
        ctx.logger.error(f" bitstream not found: {bitstream}")
        await check_result(ctx, 1, continue_run=False,
                           extra_fields={"error": "bitstream_not_found"}, trace_id=origin_tid)
        return

    # ── 1. detect PCIe BDF ────────────────────────────────────────────────
    extended_bdf = input_data.get("bus_id") or _find_xdma_bdf() or _find_xdma_pci_bdf()
    bridge = _bridge_bdf(extended_bdf) if extended_bdf else None
    ctx.logger.info(f" XDMA device: {extended_bdf}, bridge: {bridge}")

    # ── 2. flash via Vivado (JTAG — does not require PCIe) ───────────────
    cmd = (
        f"vivado -mode tcl -source {tcl_script}"
        f" -tclargs -bitstream_path {bitstream} -hw_server {hw_server_url}"
    )
    if serial:
        cmd += f" -serial {serial}"

    ctx.logger.info(f" flashing: {bitstream}")
    result = stream_run_logger(
        cmd=cmd,
        logger=ctx.logger,
        stdout_prefix="pegasus flash",
        stderr_prefix="pegasus flash",
    )

    if result.returncode != 0:
        await check_result(ctx, result.returncode, continue_run=False,
                           extra_fields={"bitstream": bitstream, "serial": serial},
                           trace_id=origin_tid)
        return

    # ── 3. PCIe remove → rescan so new bitstream's PCIe IP re-enumerates ─
    try:
        if extended_bdf:
            _pcie_disconnect(extended_bdf, bridge, ctx.logger)
        else:
            ctx.logger.info(" no xdma device found before flash, skip disconnect and try global rescan")
        _pcie_reconnect(bridge, extended_bdf, ctx.logger)
        _reload_xdma(ctx.logger)
        _wait_ddr_calibration(hw_server_url, serial, ddr_settle_sec, mig_status_tcl, ctx.logger)
    except Exception as e:
        ctx.logger.error(f" post-flash PCIe/XDMA bring-up failed: {e}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "post_flash_xdma_recover_failed", "detail": str(e)},
            trace_id=origin_tid,
        )
        return

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={"bitstream": bitstream, "serial": serial},
        trace_id=origin_tid,
    )
