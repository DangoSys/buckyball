import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API))

PEBBLE_CFG = "sims.verilator.BuckyballPebbleVerilatorConfig"


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
    mod = load_step("regression/03_eval_area_power_api.step.py")
    ctx = FakeCtx()
    resp = run(mod.handler(SimpleNamespace(body={"config": PEBBLE_CFG}), ctx))
    assert resp.status == 400
    assert "chip" in resp.body["error"]
    assert ctx.enqueued == []


def test_api_rejects_config():
    mod = load_step("regression/03_eval_area_power_api.step.py")
    ctx = FakeCtx()
    resp = run(mod.handler(SimpleNamespace(body={"chip": "pebble", "config": PEBBLE_CFG}), ctx))
    assert resp.status == 400
    assert "config" in resp.body["error"]
    assert ctx.enqueued == []


def test_api_missing_tapeout():
    mod = load_step("regression/03_eval_area_power_api.step.py")
    ctx = FakeCtx()
    resp = run(
        mod.handler(
            SimpleNamespace(body={"chip": "toy"}),
            ctx,
        )
    )
    assert resp.status == 400
    assert "tapeout" in resp.body["error"]
    assert ctx.enqueued == []


def test_api_invalid_top():
    mod = load_step("regression/03_eval_area_power_api.step.py")
    ctx = FakeCtx()
    resp = run(
        mod.handler(
            SimpleNamespace(
                body={"chip": "pebble", "top": "not a module"}
            ),
            ctx,
        )
    )
    assert resp.status == 400
    assert "top" in resp.body["error"]
    assert ctx.enqueued == []


def test_api_enqueues_regression_event():
    mod = load_step("regression/03_eval_area_power_api.step.py")
    ctx = FakeCtx("tid-api")
    resp = run(
        mod.handler(
            SimpleNamespace(body={"chip": "pebble"}),
            ctx,
        )
    )
    assert resp.status == 202
    assert resp.body == {"trace_id": "tid-api"}
    assert len(ctx.enqueued) == 1
    item = ctx.enqueued[0]
    assert item["topic"] == "regression.eval-area-power"
    assert item["data"]["chip"] == "pebble"
    assert "config" not in item["data"]
    assert item["data"]["_trace_id"] == "tid-api"
    assert "from_regression_area_power" not in item["data"]
    assert "from_area_workflow" not in item["data"]


def test_api_forward_top():
    mod = load_step("regression/03_eval_area_power_api.step.py")
    ctx = FakeCtx("tid-api")
    resp = run(
        mod.handler(
            SimpleNamespace(
                body={"chip": "pebble", "top": "DigitalTop"}
            ),
            ctx,
        )
    )
    assert resp.status == 202
    assert ctx.enqueued[0]["data"]["top"] == "DigitalTop"


def test_event_missing_chip(tmp_path, monkeypatch):
    mod = load_step("regression/03_eval_area_power_event.step.py")
    monkeypatch.setattr(mod, "get_buckyball_path", lambda: str(tmp_path))
    ctx = FakeCtx()
    run(mod.handler({"config": PEBBLE_CFG, "_trace_id": "tid-origin"}, ctx))
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["error"] == "missing_chip"
    assert ctx.enqueued == []


def test_event_missing_tapeout_without_dir(tmp_path, monkeypatch):
    mod = load_step("regression/03_eval_area_power_event.step.py")
    monkeypatch.setattr(mod, "get_buckyball_path", lambda: str(tmp_path))
    ctx = FakeCtx()
    run(mod.handler({"chip": "pebble", "_trace_id": "tid-origin"}, ctx))
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["error"] == "missing_tapeout"
    assert ctx.enqueued == []


def test_event_missing_tapeout(tmp_path, monkeypatch):
    _write_pebble_chip_toml(tmp_path)
    mod = load_step("regression/03_eval_area_power_event.step.py")
    monkeypatch.setattr(mod, "get_buckyball_path", lambda: str(tmp_path))
    ctx = FakeCtx()
    run(
        mod.handler(
            {"chip": "pebble", "_trace_id": "tid-origin"},
            ctx,
        )
    )
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["error"] == "missing_tapeout"
    assert ctx.enqueued == []


def _write_pebble_chip_toml(root: Path) -> Path:
    chip_dir = root / "examples" / "chips" / "pebble"
    chip_dir.mkdir(parents=True, exist_ok=True)
    (chip_dir / "chip.toml").write_text(
        '[chip]\nverilatorConfig = "sims.verilator.BuckyballPebbleVerilatorConfig"\n',
        encoding="utf-8",
    )
    return chip_dir


def _prep_event_tapeout(tmp_path):
    chip_dir = _write_pebble_chip_toml(tmp_path)
    tapeout = chip_dir / "tapeout"
    tapeout.mkdir()
    return tapeout


def test_event_sets_processing_and_enqueues_dc_verilog(tmp_path, monkeypatch):
    _prep_event_tapeout(tmp_path)
    mod = load_step("regression/03_eval_area_power_event.step.py")
    monkeypatch.setattr(mod, "get_buckyball_path", lambda: str(tmp_path))
    ctx = FakeCtx("tid-evt")
    run(
        mod.handler(
            {
                "chip": "pebble",
                "top": "DigitalTop",
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
    assert item["topic"] == "dc.verilog"
    data = item["data"]
    assert data["from_area_workflow"] is True
    assert data["from_regression_area_power"] is True
    assert "config" not in data
    assert data["top"] == "DigitalTop"
    assert "output_dir" not in data
    assert "analysis_dir" not in data
    assert data["_trace_id"] == "tid-origin"


def _patch_area(mod, monkeypatch, tmp_path, analysis_dir, rpt_text=None, rc=0):
    monkeypatch.setattr(mod, "get_buckyball_path", lambda: str(tmp_path))
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/bin/dc_shell")
    monkeypatch.setattr(
        mod,
        "get_tapeout_contract",
        lambda *a, **k: SimpleNamespace(
            chip="pebble",
            clock_port="clk",
            clock_period_ns=10.0,
            dc_script=str(tmp_path / "dc.tcl"),
            root=tmp_path / "tapeout",
        ),
    )
    monkeypatch.setattr(mod, "technology_settings", lambda: {})
    monkeypatch.setattr(mod, "write_run_tcl", lambda *a, **k: tmp_path / "run.tcl")

    async def fake_stream(**k):
        if rpt_text is not None:
            rpt = Path(analysis_dir) / "reports" / "area.rpt"
            rpt.parent.mkdir(parents=True, exist_ok=True)
            rpt.write_text(rpt_text)
        return SimpleNamespace(returncode=rc)

    monkeypatch.setattr(mod, "stream_run_logger_async", fake_stream)


def _area_input(tmp_path, analysis_dir, **extra):
    src = tmp_path / "dc_sources.list"
    src.write_text("a.sv\n")
    data = {
        "source_list": str(src),
        "analysis_dir": str(analysis_dir),
        "chip": "pebble",
        "top": "DigitalTop",
        "_trace_id": "tid-origin",
    }
    data.update(extra)
    return data


def test_area_without_flag_skips_metrics(tmp_path, monkeypatch):
    analysis_dir = tmp_path / "analysis"
    mod = load_step("dc/03_area_event.step.py")
    _patch_area(
        mod, monkeypatch, tmp_path, analysis_dir,
        rpt_text="Total cell area:           1000000.0000\n",
    )
    ctx = FakeCtx()
    run(mod.handler(_area_input(tmp_path, analysis_dir), ctx))
    success = ctx.state.store[("tid-origin", "success")]["body"]
    assert success["success"] is True
    assert "area" not in success
    assert "freq" not in success
    assert ctx.enqueued == []
    assert not (tmp_path / "chipcrowd-eval-result.json").exists()


def test_area_without_flag_enqueues_sim_for_power(tmp_path, monkeypatch):
    analysis_dir = tmp_path / "analysis"
    mod = load_step("dc/03_area_event.step.py")
    _patch_area(mod, monkeypatch, tmp_path, analysis_dir, rpt_text="x")
    ctx = FakeCtx()
    run(
        mod.handler(
            _area_input(tmp_path, analysis_dir, from_power_workflow=True),
            ctx,
        )
    )
    success = ctx.state.store[("tid-origin", "success")]["body"]
    assert success["success"] is True
    assert len(ctx.enqueued) == 1
    assert ctx.enqueued[0]["topic"] == "dc.sim"
    assert not (tmp_path / "chipcrowd-eval-result.json").exists()


def test_area_flag_merges_area_freq(tmp_path, monkeypatch):
    analysis_dir = tmp_path / "analysis"
    mod = load_step("dc/03_area_event.step.py")
    _patch_area(
        mod, monkeypatch, tmp_path, analysis_dir,
        rpt_text="Total cell area:           1000000.0000\n",
    )
    ctx = FakeCtx()
    run(
        mod.handler(
            _area_input(
                tmp_path, analysis_dir,
                from_area_workflow=True,
                from_regression_area_power=True,
            ),
            ctx,
        )
    )
    success = ctx.state.store[("tid-origin", "success")]["body"]
    assert success["success"] is True
    assert success["area"] == 1.0
    assert success["freq"] == 100.0
    assert ctx.enqueued == []
    data = json.loads((tmp_path / "chipcrowd-eval-result.json").read_text())
    assert data["area"] == 1.0
    assert data["freq"] == 100.0


def test_area_flag_does_not_enqueue_sim(tmp_path, monkeypatch):
    analysis_dir = tmp_path / "analysis"
    mod = load_step("dc/03_area_event.step.py")
    _patch_area(
        mod, monkeypatch, tmp_path, analysis_dir,
        rpt_text="Total cell area:           1000000.0000\n",
    )
    ctx = FakeCtx()
    run(
        mod.handler(
            _area_input(
                tmp_path, analysis_dir,
                from_power_workflow=True,
                from_regression_area_power=True,
            ),
            ctx,
        )
    )
    assert ctx.enqueued == []
    assert (tmp_path / "chipcrowd-eval-result.json").is_file()


def test_area_flag_missing_rpt_fails(tmp_path, monkeypatch):
    analysis_dir = tmp_path / "analysis"
    mod = load_step("dc/03_area_event.step.py")
    _patch_area(mod, monkeypatch, tmp_path, analysis_dir, rpt_text=None)
    ctx = FakeCtx()
    run(
        mod.handler(
            _area_input(tmp_path, analysis_dir, from_regression_area_power=True),
            ctx,
        )
    )
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["failure"] is True
    assert "missing area.rpt" in failure["error"]
    assert ctx.enqueued == []
    assert not (tmp_path / "chipcrowd-eval-result.json").exists()


def test_area_flag_unparsable_rpt_fails(tmp_path, monkeypatch):
    analysis_dir = tmp_path / "analysis"
    mod = load_step("dc/03_area_event.step.py")
    _patch_area(mod, monkeypatch, tmp_path, analysis_dir, rpt_text="no area here\n")
    ctx = FakeCtx()
    run(
        mod.handler(
            _area_input(tmp_path, analysis_dir, from_regression_area_power=True),
            ctx,
        )
    )
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["failure"] is True
    assert failure["error"] == "Total cell area not found"
    assert ctx.enqueued == []
    assert not (tmp_path / "chipcrowd-eval-result.json").exists()
