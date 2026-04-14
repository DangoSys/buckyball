"""
pegasus flashbitstream event handler

Sequence:
  1. Detect XDMA PCIe BDF (auto-scan /sys/bus/pci/devices for xdma driver)
  2. PCIe disconnect: clear SERR + fatal-error bits, remove device
  3. Flash via Vivado + program_fpga.tcl
  4. PCIe reconnect: rescan, re-enable memory-mapped transfers
"""
import os
import sys
import glob
import time
import subprocess

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


def _find_xdma_bdf() -> str | None:
    """Return the extended BDF (e.g. '0000:65:00.0') of the first xdma device."""
    for entry in os.listdir(PCI_DEVICES):
        driver_link = os.path.join(PCI_DEVICES, entry, "driver")
        if not os.path.islink(driver_link):
            continue
        driver_name = os.path.basename(os.readlink(driver_link))
        if driver_name == "xdma":
            return entry   # e.g. '0000:65:00.0'
    return None


def _bridge_bdf(extended_bdf: str) -> str | None:
    """Return the bridge BDF for a given extended device BDF."""
    dev_path = os.path.realpath(os.path.join(PCI_DEVICES, extended_bdf))
    bridge_path = os.path.dirname(dev_path)
    bridge_bdf = os.path.basename(bridge_path)
    if bridge_bdf.startswith("0000:"):
        return bridge_bdf
    return None


def _setpci(bdf: str, reg: str, val: str) -> bool:
    r = subprocess.run(["setpci", "-s", bdf, f"{reg}={val}"],
                       capture_output=True)
    return r.returncode == 0


def _pcie_disconnect(extended_bdf: str, bridge: str | None, logger) -> None:
    """Clear SERR/fatal-error bits and remove device from PCIe bus."""
    if bridge:
        logger.info(f"[pegasus] clearing SERR bit on bridge {bridge}")
        _setpci(bridge, "COMMAND", "0000:0100")
        time.sleep(0.5)
        logger.info(f"[pegasus] clearing fatal-error bit on bridge {bridge}")
        _setpci(bridge, "CAP_EXP+8.w", "0000:0004")
        time.sleep(0.5)
    remove_path = os.path.join(PCI_DEVICES, extended_bdf, "remove")
    if os.path.exists(remove_path):
        logger.info(f"[pegasus] removing PCIe device {extended_bdf}")
        try:
            with open(remove_path, "w") as f:
                f.write("1\n")
            time.sleep(2)
        except OSError as e:
            logger.warning(f"[pegasus] remove write failed: {e}")


def _pcie_reconnect(bridge: str | None, device_bdf: str | None, logger) -> None:
    """Rescan PCIe bus and re-enable memory-mapped transfers."""
    if bridge:
        rescan_path = os.path.join(PCI_DEVICES, bridge, "rescan")
        if os.path.exists(rescan_path):
            logger.info(f"[pegasus] rescanning bridge {bridge}")
            try:
                with open(rescan_path, "w") as f:
                    f.write("1\n")
                time.sleep(1)
            except OSError:
                pass
    global_rescan = "/sys/bus/pci/rescan"
    logger.info("[pegasus] global PCIe rescan")
    try:
        with open(global_rescan, "w") as f:
            f.write("1\n")
        time.sleep(2)
    except OSError:
        pass
    # re-enable memory-mapped transfers on new BDF (re-detect after rescan)
    new_bdf = _find_xdma_bdf()
    if new_bdf:
        logger.info(f"[pegasus] enabling MEM transfers on {new_bdf}")
        _setpci(new_bdf, "COMMAND", "0x02")


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    bitstream = input_data.get(
        "bitstream",
        os.path.join(bbdir, "thirdparty", "pegasus", "vivado", "build", "PegasusTop.bit"),
    )
    serial = input_data.get("serial", "")   # empty = auto (first target)
    tcl_script = os.path.join(
        bbdir, "thirdparty", "pegasus", "vivado", "scripts", "program_fpga.tcl"
    )

    if not os.path.exists(bitstream):
        ctx.logger.error(f"[pegasus] bitstream not found: {bitstream}")
        await check_result(ctx, 1, continue_run=False,
                           extra_fields={"error": "bitstream_not_found"}, trace_id=origin_tid)
        return

    # ── 1. detect PCIe BDF ────────────────────────────────────────────────
    extended_bdf = input_data.get("bus_id") or _find_xdma_bdf()
    bridge = _bridge_bdf(extended_bdf) if extended_bdf else None
    ctx.logger.info(f"[pegasus] XDMA device: {extended_bdf}, bridge: {bridge}")

    # ── 2. PCIe disconnect ────────────────────────────────────────────────
    if extended_bdf:
        _pcie_disconnect(extended_bdf, bridge, ctx.logger)
    else:
        ctx.logger.warning("[pegasus] no xdma device found, skipping PCIe disconnect")

    # ── 3. flash via Vivado ───────────────────────────────────────────────
    cmd = f"vivado -mode batch -source {tcl_script} -tclargs -bitstream_path {bitstream}"
    if serial:
        cmd += f" -serial {serial}"

    ctx.logger.info(f"[pegasus] flashing: {bitstream}")
    result = stream_run_logger(
        cmd=cmd,
        logger=ctx.logger,
        stdout_prefix="pegasus flash",
        stderr_prefix="pegasus flash",
    )

    # ── 4. PCIe reconnect (regardless of flash result) ───────────────────
    if extended_bdf:
        _pcie_reconnect(bridge, extended_bdf, ctx.logger)

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={"bitstream": bitstream, "serial": serial},
        trace_id=origin_tid,
    )
