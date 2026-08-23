import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API))


def load_step(rel):
    path = API / "steps" / rel
    name = path.name.replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load step: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeCtx:
    def __init__(self, trace_id="tid-api"):
        self.trace_id = trace_id
        self.enqueued = []
        self.logger = SimpleNamespace(info=lambda *a, **k: None,
                                      error=lambda *a, **k: None)

    async def enqueue(self, item):
        self.enqueued.append(item)


def run(coro):
    return asyncio.run(coro)


def test_api_missing_chip(tmp_path):
    bs = tmp_path / "bitstream.bit"
    bs.write_text("x")
    mod = load_step("regression/02_eval_performance_api.step.py")
    ctx = FakeCtx()
    resp = run(mod.handler(SimpleNamespace(body={"bitstream": str(bs)}), ctx))
    assert resp.status == 400
    assert "chip" in resp.body["error"]
    assert ctx.enqueued == []


def test_api_missing_bitstream():
    mod = load_step("regression/02_eval_performance_api.step.py")
    ctx = FakeCtx()
    resp = run(mod.handler(SimpleNamespace(body={"chip": "pebble"}), ctx))
    assert resp.status == 400
    assert "bitstream" in resp.body["error"]
    assert ctx.enqueued == []


def test_api_bitstream_not_found():
    mod = load_step("regression/02_eval_performance_api.step.py")
    ctx = FakeCtx()
    resp = run(
        mod.handler(
            SimpleNamespace(
                body={"chip": "pebble", "bitstream": "/no/such/bitstream.bit"},
            ),
            ctx,
        )
    )
    assert resp.status == 400
    assert "not found" in resp.body["error"]
    assert ctx.enqueued == []


def test_api_enqueues_regression_event(tmp_path):
    bs = tmp_path / "bitstream.bit"
    bs.write_text("x")
    mod = load_step("regression/02_eval_performance_api.step.py")
    ctx = FakeCtx("tid-api")
    resp = run(
        mod.handler(
            SimpleNamespace(body={"chip": "pebble", "bitstream": str(bs)}), ctx)
        )
    assert resp.status == 202
    assert resp.body == {"trace_id": "tid-api"}
    assert len(ctx.enqueued) == 1
    item = ctx.enqueued[0]
    assert item["topic"] == "regression.eval-performance"
    assert item["data"]["chip"] == "pebble"
    assert item["data"]["bitstream"] == str(bs)
    assert item["data"]["_trace_id"] == "tid-api"
    assert "config" not in item["data"]


def test_api_forward_config(tmp_path):
    bs = tmp_path / "bitstream.bit"
    bs.write_text("x")
    mod = load_step("regression/02_eval_performance_api.step.py")
    ctx = FakeCtx("tid-api")
    resp = run(
        mod.handler(
            SimpleNamespace(
                body={
                    "chip": "pebble",
                    "bitstream": str(bs),
                    "config": "sims.p2e.P2EPebbleLinuxConfig",
                }
            ),
            ctx,
        )
    )
    assert resp.status == 202
    assert ctx.enqueued[0]["data"]["config"] == "sims.p2e.P2EPebbleLinuxConfig"
