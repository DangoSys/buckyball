import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API))


def load_step(rel: str):
    path = API / "steps" / rel
    name = path.name.replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load step: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeState:
    def __init__(self):
        self.store = {}

    async def set(self, scope, key, val):
        self.store[(scope, key)] = val


class FakeCtx:
    def __init__(self, trace_id="tid-origin"):
        self.trace_id = trace_id
        self.enqueued = []
        self.state = FakeState()
        self.logger = SimpleNamespace(
            info=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )

    async def enqueue(self, item):
        self.enqueued.append(item)


def run(coro):
    return asyncio.run(coro)


def test_api_missing_chip():
    mod = load_step("regression/00_buildbitstream_api.step.py")
    ctx = FakeCtx()
    resp = run(mod.handler(SimpleNamespace(body={"config": "sims.p2e.P2EPebbleLinuxConfig"}), ctx))
    assert resp.status == 400
    assert "chip" in resp.body["error"]
    assert ctx.enqueued == []


def test_api_missing_config():
    mod = load_step("regression/00_buildbitstream_api.step.py")
    ctx = FakeCtx()
    resp = run(mod.handler(SimpleNamespace(body={"chip": "pebble"}), ctx))
    assert resp.status == 400
    assert "config" in resp.body["error"]
    assert ctx.enqueued == []


def test_api_enqueues_regression_event():
    mod = load_step("regression/00_buildbitstream_api.step.py")
    ctx = FakeCtx("tid-api")
    resp = run(
        mod.handler(
            SimpleNamespace(
                body={
                    "chip": "pebble",
                    "config": "sims.p2e.P2EPebbleLinuxConfig",
                    "output_dir": "/tmp/bs",
                }
            ),
            ctx,
        )
    )
    assert resp.status == 202
    assert resp.body == {"trace_id": "tid-api"}
    assert len(ctx.enqueued) == 1
    item = ctx.enqueued[0]
    assert item["topic"] == "regression.buildbitstream"
    assert item["data"]["chip"] == "pebble"
    assert item["data"]["config"] == "sims.p2e.P2EPebbleLinuxConfig"
    assert item["data"]["output_dir"] == "/tmp/bs"
    assert item["data"]["_trace_id"] == "tid-api"
    assert "from_regression_buildbitstream" not in item["data"]


def test_event_sets_processing_and_enqueues_verilog():
    mod = load_step("regression/00_buildbitstream_event.step.py")
    ctx = FakeCtx("tid-evt")
    run(
        mod.handler(
            {
                "chip": "pebble",
                "config": "sims.p2e.P2EPebbleLinuxConfig",
                "output_dir": "/tmp/bs",
                "_trace_id": "tid-origin",
            },
            ctx,
        )
    )
    assert ctx.state.store[("tid-origin", "processing")]["processing"] is True
    assert ("tid-origin", "success") not in ctx.state.store
    assert ("tid-origin", "failure") not in ctx.state.store
    assert len(ctx.enqueued) == 1
    item = ctx.enqueued[0]
    assert item["topic"] == "bebop.p2e.verilog"
    data = item["data"]
    assert data["from_regression_buildbitstream"] is True
    assert data["chip"] == "pebble"
    assert data["config"] == "sims.p2e.P2EPebbleLinuxConfig"
    assert data["_trace_id"] == "tid-origin"
    assert data["build_dir"] == "/tmp/bs"
    assert "output_dir" not in data


def _patch_verilog(mod, monkeypatch, rc=0):
    monkeypatch.setattr(mod, "get_buckyball_path", lambda: "/tmp/bb")
    monkeypatch.setattr(
        mod, "get_verilator_build_dir", lambda bb, cfg, out: out or "/tmp/bb/arch/build/cfg"
    )
    monkeypatch.setattr(mod, "cleanup_strays", lambda *a, **k: None)
    monkeypatch.setattr(mod, "normalize_p2e_timescale", lambda *a, **k: None)
    monkeypatch.setattr(mod.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(
        mod, "stream_run_logger", lambda **k: SimpleNamespace(returncode=rc)
    )


def test_verilog_without_flag_finalizes(monkeypatch):
    mod = load_step("mill/07_bebop_p2e_verilog_event.step.py")
    _patch_verilog(mod, monkeypatch, rc=0)
    ctx = FakeCtx()
    run(mod.handler({"config": "sims.p2e.P2EPebbleLinuxConfig", "_trace_id": "tid-origin"}, ctx))
    assert ctx.enqueued == []
    success = ctx.state.store[("tid-origin", "success")]
    assert success["body"]["success"] is True
    assert success["body"]["task"] == "verilog"
    assert ("tid-origin", "processing") not in ctx.state.store


def test_verilog_with_flag_continues_and_enqueues(monkeypatch):
    mod = load_step("mill/07_bebop_p2e_verilog_event.step.py")
    _patch_verilog(mod, monkeypatch, rc=0)
    ctx = FakeCtx()
    run(
        mod.handler(
            {
                "chip": "pebble",
                "config": "sims.p2e.P2EPebbleLinuxConfig",
                "from_regression_buildbitstream": True,
                "build_dir": "/tmp/bs",
                "_trace_id": "tid-origin",
            },
            ctx,
        )
    )
    proc = ctx.state.store[("tid-origin", "processing")]
    assert proc["processing"] is True
    assert ("tid-origin", "success") not in ctx.state.store
    assert len(ctx.enqueued) == 1
    item = ctx.enqueued[0]
    assert item["topic"] == "bebop.p2e.buildbitstream"
    data = item["data"]
    assert data["from_regression_buildbitstream"] is True
    assert data["chip"] == "pebble"
    assert data["vsrc_dir"] == "/tmp/bb/arch/build/cfg"
    assert data["build_dir"] == "/tmp/bs"
    assert data["_trace_id"] == "tid-origin"


def test_verilog_failure_with_flag_does_not_enqueue(monkeypatch):
    mod = load_step("mill/07_bebop_p2e_verilog_event.step.py")
    _patch_verilog(mod, monkeypatch, rc=7)
    ctx = FakeCtx()
    run(
        mod.handler(
            {
                "config": "sims.p2e.P2EPebbleLinuxConfig",
                "from_regression_buildbitstream": True,
                "_trace_id": "tid-origin",
            },
            ctx,
        )
    )
    assert ctx.enqueued == []
    failure = ctx.state.store[("tid-origin", "failure")]
    assert failure["body"]["failure"] is True
    assert failure["body"]["returncode"] == 7
    assert ("tid-origin", "success") not in ctx.state.store


def _patch_buildbitstream(mod, monkeypatch):
    monkeypatch.setattr(mod, "get_buckyball_path", lambda: "/tmp/bb")
    monkeypatch.setattr(mod, "get_verilator_build_dir", lambda bb, cfg, vsrc: vsrc or "/tmp/vsrc")
    monkeypatch.setattr(mod.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(mod.os.path, "exists", lambda p: True)
    monkeypatch.setattr(mod.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(mod, "stream_run_logger", lambda **k: SimpleNamespace(returncode=0))


def test_buildbitstream_without_flag_keeps_relative_bitstream(monkeypatch):
    mod = load_step("bebop/p2e/02_buildbitstream_event.step.py")
    _patch_buildbitstream(mod, monkeypatch)
    ctx = FakeCtx()
    run(
        mod.handler(
            {
                "config": "sims.p2e.P2EPebbleLinuxConfig",
                "vsrc_dir": "/tmp/vsrc",
                "output_dir": "rel-bs",
                "_trace_id": "tid-origin",
            },
            ctx,
        )
    )
    success = ctx.state.store[("tid-origin", "success")]["body"]
    assert success["success"] is True
    assert success["bitstream"] == os.path.join("rel-bs", "fpgaCompDir", "bitstream.bit")


def test_buildbitstream_with_flag_failure_keeps_relative_bitstream(monkeypatch):
    mod = load_step("bebop/p2e/02_buildbitstream_event.step.py")
    _patch_buildbitstream(mod, monkeypatch)
    monkeypatch.setattr(mod, "stream_run_logger", lambda **k: SimpleNamespace(returncode=1))
    ctx = FakeCtx()
    run(
        mod.handler(
            {
                "config": "sims.p2e.P2EPebbleLinuxConfig",
                "vsrc_dir": "/tmp/vsrc",
                "output_dir": "rel-bs",
                "from_regression_buildbitstream": True,
                "_trace_id": "tid-origin",
            },
            ctx,
        )
    )
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["failure"] is True
    assert failure["bitstream"] == os.path.join("rel-bs", "fpgaCompDir", "bitstream.bit")
    assert not os.path.isabs(failure["bitstream"])


def test_buildbitstream_with_flag_absolutes_bitstream(monkeypatch):
    mod = load_step("bebop/p2e/02_buildbitstream_event.step.py")
    _patch_buildbitstream(mod, monkeypatch)
    ctx = FakeCtx()
    run(
        mod.handler(
            {
                "config": "sims.p2e.P2EPebbleLinuxConfig",
                "vsrc_dir": "/tmp/vsrc",
                "output_dir": "rel-bs",
                "from_regression_buildbitstream": True,
                "_trace_id": "tid-origin",
            },
            ctx,
        )
    )
    success = ctx.state.store[("tid-origin", "success")]["body"]
    assert success["success"] is True
    assert success["bitstream"] == os.path.abspath(
        os.path.join("rel-bs", "fpgaCompDir", "bitstream.bit")
    )
    assert os.path.isabs(success["bitstream"])
