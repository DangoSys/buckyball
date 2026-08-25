import asyncio
import importlib.util
import json
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


def test_api_missing_chip(tmp_path):
    bs = tmp_path / "bitstream.bit"
    bs.write_text("x")
    mod = load_step("regression/01_check_api.step.py")
    ctx = FakeCtx()
    resp = run(mod.handler(SimpleNamespace(body={"bitstream": str(bs)}), ctx))
    assert resp.status == 400
    assert "chip" in resp.body["error"]
    assert ctx.enqueued == []


def test_api_missing_bitstream():
    mod = load_step("regression/01_check_api.step.py")
    ctx = FakeCtx()
    resp = run(mod.handler(SimpleNamespace(body={"chip": "pebble"}), ctx))
    assert resp.status == 400
    assert "bitstream" in resp.body["error"]
    assert ctx.enqueued == []


def test_api_bitstream_not_found():
    mod = load_step("regression/01_check_api.step.py")
    ctx = FakeCtx()
    resp = run(
        mod.handler(
            SimpleNamespace(body={"chip": "pebble", "bitstream": "/no/such/bitstream.bit"}),
            ctx,
        )
    )
    assert resp.status == 400
    assert "not found" in resp.body["error"]
    assert ctx.enqueued == []


def test_api_enqueues_regression_event(tmp_path):
    bs = tmp_path / "bitstream.bit"
    bs.write_text("x")
    mod = load_step("regression/01_check_api.step.py")
    ctx = FakeCtx("tid-api")
    resp = run(
        mod.handler(
            SimpleNamespace(body={"chip": "pebble", "bitstream": str(bs)}),
            ctx,
        )
    )
    assert resp.status == 202
    assert resp.body == {"trace_id": "tid-api"}
    assert len(ctx.enqueued) == 1
    item = ctx.enqueued[0]
    assert item["topic"] == "regression.check"
    assert item["data"]["chip"] == "pebble"
    assert item["data"]["bitstream"] == str(bs)
    assert item["data"]["_trace_id"] == "tid-api"
    assert "from_regression_check" not in item["data"]


def test_event_sets_processing_and_enqueues_batch(tmp_path):
    bs = tmp_path / "bitstream.bit"
    bs.write_text("x")
    mod = load_step("regression/01_check_event.step.py")
    ctx = FakeCtx("tid-evt")
    run(
        mod.handler(
            {
                "chip": "pebble",
                "bitstream": str(bs),
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
    assert item["topic"] == "bebop.p2e.batch"
    data = item["data"]
    assert data["from_regression_check"] is True
    assert data["test"] == "pk-tests"
    assert data["chip"] == "pebble"
    assert data["bitstream"] == str(bs)
    assert data["_trace_id"] == "tid-origin"


def _prep_batch_case(tmp_path):
    build = tmp_path / "case"
    fpga = build / "fpgaCompDir"
    fpga.mkdir(parents=True)
    bs = fpga / "bitstream.bit"
    bs.write_text("x")
    rt = build / "vvacDir" / "runtimeDir"
    rt.mkdir(parents=True)
    (rt / "rtcfg").write_text("x")
    lib = rt / "lib" / "lib_arm"
    lib.mkdir(parents=True)
    (lib / "libvCtb.so").write_text("x")
    vsrc = tmp_path / "vsrc"
    vsrc.mkdir()
    return str(bs), str(vsrc)


def _patch_batch(mod, monkeypatch, tmp_path, vsrc, stdout="", stderr="", rc=0):
    monkeypatch.setattr(mod, "get_buckyball_path", lambda: str(tmp_path))
    monkeypatch.setattr(mod, "rtl_dir", lambda *a, **k: vsrc)
    monkeypatch.setattr(mod, "regression_workload_toml", lambda *a, **k: "/tmp/workloads-pk.toml")

    def fake_stream(**k):
        prefix = k.get("stdout_prefix", "")
        if prefix == "bebop p2e batch":
            return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "stream_run_logger", fake_stream)


def test_batch_without_flag_skips_accuracy(tmp_path, monkeypatch):
    bs, vsrc = _prep_batch_case(tmp_path)
    mod = load_step("bebop/p2e/04_batch_event.step.py")
    _patch_batch(
        mod, monkeypatch, tmp_path, vsrc,
        stdout="Summary [ 1.2s] 4 tests run: 3 passed, 1 failed",
    )
    ctx = FakeCtx()
    run(
        mod.handler(
            {
                "chip": "pebble",
                "bitstream": bs,
                "test": "pk-tests",
                "_trace_id": "tid-origin",
            },
            ctx,
        )
    )
    success = ctx.state.store[("tid-origin", "success")]["body"]
    assert success["success"] is True
    assert "accuracy" not in success
    assert not (tmp_path / "chipcrowd-eval-result.json").exists()


def test_batch_flag_merges_accuracy(tmp_path, monkeypatch):
    bs, vsrc = _prep_batch_case(tmp_path)
    mod = load_step("bebop/p2e/04_batch_event.step.py")
    _patch_batch(
        mod, monkeypatch, tmp_path, vsrc,
        stdout="Summary [ 1.2s] 4 tests run: 3 passed, 1 failed",
        rc=1,
    )
    ctx = FakeCtx()
    run(
        mod.handler(
            {
                "chip": "pebble",
                "bitstream": bs,
                "test": "pk-tests",
                "from_regression_check": True,
                "_trace_id": "tid-origin",
            },
            ctx,
        )
    )
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["failure"] is True
    assert failure["returncode"] == 1
    assert failure["accuracy"] == 0.75
    data = json.loads((tmp_path / "chipcrowd-eval-result.json").read_text())
    assert data["accuracy"] == 0.75


def test_batch_flag_unparsable_fails(tmp_path, monkeypatch):
    bs, vsrc = _prep_batch_case(tmp_path)
    mod = load_step("bebop/p2e/04_batch_event.step.py")
    _patch_batch(mod, monkeypatch, tmp_path, vsrc, stdout="no summary here", rc=0)
    ctx = FakeCtx()
    run(
        mod.handler(
            {
                "chip": "pebble",
                "bitstream": bs,
                "test": "pk-tests",
                "from_regression_check": True,
                "_trace_id": "tid-origin",
            },
            ctx,
        )
    )
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["failure"] is True
    assert failure["error"] == "accuracy_unparsable"
    assert not (tmp_path / "chipcrowd-eval-result.json").exists()
