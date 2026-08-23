import asyncio
import importlib.util
import json
import os
import re
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


class FakeState:
    def __init__(self):
        self.store = {}

    async def set(self, scope, key, val):
        self.store[(scope, key)] = val


class FakeCtx:
    def __init__(self, trace_id="tid-evt"):
        self.trace_id = trace_id
        self.enqueued = []
        self.state = FakeState()
        self.logger = SimpleNamespace(info=lambda *a, **k: None,
                                      error=lambda *a, **k: None)

    async def enqueue(self, item):
        self.enqueued.append(item)


def run(coro):
    return asyncio.run(coro)


def _make_bitstream(tmp_path):
    build = tmp_path / "case"
    fpga = build / "fpgaCompDir"
    fpga.mkdir(parents=True)
    bs = fpga / "bitstream.bit"
    bs.write_text("x")
    rt = build / "vvacDir" / "runtimeDir"
    lib = rt / "lib" / "lib_arm"
    lib.mkdir(parents=True)
    (lib / "libvCtb.so").write_text("x")
    (rt / "rtcfg").write_text("x")
    vsrc = tmp_path / "vsrc"
    vsrc.mkdir()
    return str(bs), str(vsrc), str(build)


def _patch_paths(mod, monkeypatch, tmp_path, vsrc):
    monkeypatch.setattr(mod, "get_buckyball_path", lambda: str(tmp_path))
    monkeypatch.setattr(mod, "resolve_chip_compiler_core",
                        lambda bb, chip: SimpleNamespace(name="pebbleCore"))
    monkeypatch.setattr(mod, "get_verilator_build_dir",
                        lambda bb, cfg, out: out or str(vsrc))


def _fake_stream_factory(tmp_path):
    def fake(**k):
        prefix = k.get("stdout_prefix", "")
        cmd = k.get("cmd", "")
        rc = 0
        if prefix.startswith("workload build"):
            src = (tmp_path / "bb-tests" / "output" / "workloads" / "src"
                   / "ModelTest" / "e2e" / "models" / "archs" / "buckyball"
                   / "pebble" / "LeNet")
            src.mkdir(parents=True, exist_ok=True)
            (src / "buddy-buckyball-lenet-run").write_text("x")
            bld = (tmp_path / "bb-tests" / "build" / "workloads" / "src"
                   / "ModelTest" / "e2e" / "models" / "archs" / "buckyball"
                   / "pebble" / "LeNet")
            bld.mkdir(parents=True, exist_ok=True)
            (bld / "subgraph0_linalg.mlir").write_text("linalg")
            (bld / "subgraph0_buckyball.mlir").write_text("buckyball")
        elif prefix.startswith("kernel build"):
            out = tmp_path / "bb-tests" / "output" / "kernel"
            out.mkdir(parents=True, exist_ok=True)
            (out / "fw_payload-lenet.bin").write_text("bin")
        elif prefix.startswith("p2e run"):
            m = re.search(r'--log-dir="([^"]+)"', cmd)
            if m:
                cycle = Path(m.group(1)) / "trace" / "cycle"
                cycle.mkdir(parents=True, exist_ok=True)
                (cycle / "trace-0.txt").write_text("start 10\nend 20\nelapsed 10\n")
        elif prefix.startswith("perfetto"):
            m = re.search(r"^(\S+) \S+ (\S+) (\S+)", cmd)
            trace_dir = Path(m.group(2)) if m else None
            if trace_dir:
                out = trace_dir / "perfetto.json"
                out.write_text(json.dumps({
                    "traceEvents": [{"ph": "X", "ts": 10, "dur": 5}]
                }))
        return SimpleNamespace(returncode=rc, stdout="", stderr="")
    return fake


def test_event_missing_bitstream(tmp_path, monkeypatch):
    mod = load_step("regression/02_eval_performance_event.step.py")
    _patch_paths(mod, monkeypatch, tmp_path, tmp_path / "vsrc")
    ctx = FakeCtx()
    run(mod.handler({"chip": "pebble", "bitstream": "/no/such.bit",
                     "_trace_id": "tid-origin"}, ctx))
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["failure"] is True
    assert failure["error"] == "bitstream_not_found"
    assert ctx.enqueued == []


def test_event_missing_models_toml(tmp_path, monkeypatch):
    bs, vsrc, _ = _make_bitstream(tmp_path)
    mod = load_step("regression/02_eval_performance_event.step.py")
    _patch_paths(mod, monkeypatch, tmp_path, vsrc)
    ctx = FakeCtx()
    run(mod.handler({"chip": "pebble", "bitstream": bs,
                     "_trace_id": "tid-origin"}, ctx))
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["failure"] is True
    assert failure["error"] == "models_toml_invalid"
    assert not (tmp_path / "chipcrowd-eval-result.json").exists()


def test_event_missing_chip(tmp_path, monkeypatch):
    bs, vsrc, _ = _make_bitstream(tmp_path)
    mod = load_step("regression/02_eval_performance_event.step.py")
    _patch_paths(mod, monkeypatch, tmp_path, vsrc)
    ctx = FakeCtx()
    run(mod.handler({"bitstream": bs, "_trace_id": "tid-origin"}, ctx))
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["error"] == "missing_chip"


def _seed_models_toml(tmp_path, models=None):
    if models is None:
        models = ["lenet"]
    d = (tmp_path / "examples" / "chips" / "pebble" / "regression" / "eval")
    d.mkdir(parents=True, exist_ok=True)
    (d / "models.toml").write_text(
        "[eval]\nmodels = " + json.dumps(models) + "\n"
    )


def _seed_perfetto_sources(tmp_path):
    toml = (tmp_path / "bb-tests" / "workloads" / "src" / "ModelTest"
            / "e2e" / "models" / "models" / "LeNet" / "trace"
            / "trace-linalg-buckyball.toml")
    toml.parent.mkdir(parents=True, exist_ok=True)
    toml.write_text("[trace]\n")


def test_event_happy_path_merges_latency(tmp_path, monkeypatch):
    bs, vsrc, _ = _make_bitstream(tmp_path)
    _seed_models_toml(tmp_path)
    _seed_perfetto_sources(tmp_path)
    mod = load_step("regression/02_eval_performance_event.step.py")
    _patch_paths(mod, monkeypatch, tmp_path, vsrc)
    monkeypatch.setattr(mod, "stream_run_logger", _fake_stream_factory(tmp_path))
    monkeypatch.setattr(mod._kernel, "bin_to_hex",
                        lambda bin_p, hex_p, base_address=0: (
                            Path(hex_p).parent.mkdir(parents=True, exist_ok=True),
                            Path(hex_p).write_text("hex"),
                            True,
                        )[-1])
    ctx = FakeCtx()
    run(mod.handler({"chip": "pebble", "bitstream": bs,
                     "_trace_id": "tid-origin"}, ctx))
    success = ctx.state.store[("tid-origin", "success")]["body"]
    assert success["success"] is True
    assert success["latency"] == 5.0
    assert len(success["models"]) == 1
    assert success["models"][0]["name"] == "lenet"
    assert success["models"][0]["latency"] == 5.0
    data = json.loads((tmp_path / "chipcrowd-eval-result.json").read_text())
    assert data["latency"] == 5.0
    assert data["models"][0]["name"] == "lenet"
    assert data["models"][0]["perfetto"].endswith("perfetto-lenet.json")


def test_event_unknown_model_fails_cleanly(tmp_path, monkeypatch):
    bs, vsrc, _ = _make_bitstream(tmp_path)
    _seed_models_toml(tmp_path, models=["not-a-real-model"])
    mod = load_step("regression/02_eval_performance_event.step.py")
    _patch_paths(mod, monkeypatch, tmp_path, vsrc)
    ctx = FakeCtx()
    run(mod.handler({"chip": "pebble", "bitstream": bs,
                     "_trace_id": "tid-origin"}, ctx))
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["failure"] is True
    assert failure["error"] == "unknown_model"
    assert failure["model"] == "not-a-real-model"
    assert not (tmp_path / "chipcrowd-eval-result.json").exists()


def test_workload_build_cmd_unknown_model_is_valueerror(tmp_path, monkeypatch):
    mod = load_step("regression/02_eval_performance_event.step.py")
    monkeypatch.setattr(mod, "resolve_chip_compiler_core",
                        lambda bb, chip: SimpleNamespace(name="pebbleCore"))
    try:
        mod._workload_build_cmd(str(tmp_path), "pebble", "not-a-real-model")
        assert False
    except ValueError as e:
        assert "Unknown model" in str(e)
    except KeyError:
        assert False, "unknown model must raise ValueError, not KeyError"


def test_event_missing_model_layout_fails_cleanly(tmp_path, monkeypatch):
    bs, vsrc, _ = _make_bitstream(tmp_path)
    _seed_models_toml(tmp_path, models=["lenet"])
    mod = load_step("regression/02_eval_performance_event.step.py")
    _patch_paths(mod, monkeypatch, tmp_path, vsrc)
    monkeypatch.setattr(mod, "MODEL_LAYOUT", {})
    ctx = FakeCtx()
    run(mod.handler({"chip": "pebble", "bitstream": bs,
                     "_trace_id": "tid-origin"}, ctx))
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["error"] == "missing_model_layout"
    assert failure["model"] == "lenet"
    assert not (tmp_path / "chipcrowd-eval-result.json").exists()


def test_event_missing_build_dir_fails_cleanly(tmp_path, monkeypatch):
    bs, vsrc, build = _make_bitstream(tmp_path)
    _seed_models_toml(tmp_path)
    _seed_perfetto_sources(tmp_path)
    mod = load_step("regression/02_eval_performance_event.step.py")
    _patch_paths(mod, monkeypatch, tmp_path, vsrc)
    monkeypatch.setattr(mod, "stream_run_logger", _fake_stream_factory(tmp_path))
    monkeypatch.setattr(mod._kernel, "bin_to_hex",
                        lambda bin_p, hex_p, base_address=0: (
                            Path(hex_p).parent.mkdir(parents=True, exist_ok=True),
                            Path(hex_p).write_text("hex"),
                            True,
                        )[-1])
    orig_isdir = os.path.isdir

    def fake_isdir(path):
        if os.path.abspath(path) == os.path.abspath(build):
            return False
        return orig_isdir(path)

    monkeypatch.setattr(mod.os.path, "isdir", fake_isdir)
    ctx = FakeCtx()
    run(mod.handler({"chip": "pebble", "bitstream": bs,
                     "_trace_id": "tid-origin"}, ctx))
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["error"] == "build_dir_not_found"
    assert failure["model"] == "lenet"
    assert failure["build_dir"] == os.path.abspath(build)
    assert not (tmp_path / "chipcrowd-eval-result.json").exists()


def test_p2e_run_cmds_missing_build_dir(tmp_path, monkeypatch):
    mod = load_step("regression/02_eval_performance_event.step.py")
    hex_dir = tmp_path / "bb-tests" / "output" / "kernel"
    hex_dir.mkdir(parents=True)
    (hex_dir / "fw.hex").write_text("x")
    fpga = tmp_path / "case" / "fpgaCompDir"
    fpga.mkdir(parents=True)
    bs = fpga / "bitstream.bit"
    bs.write_text("x")
    build = str(tmp_path / "case")
    orig_isdir = os.path.isdir

    def fake_isdir(path):
        if os.path.abspath(path) == os.path.abspath(build):
            return False
        return orig_isdir(path)

    monkeypatch.setattr(mod.os.path, "isdir", fake_isdir)
    try:
        mod._p2e_run_cmds(str(tmp_path), str(bs), "fw", None, {})
        assert False
    except FileNotFoundError as e:
        assert "P2E build case not found" in str(e)
        assert build in str(e)


def test_event_workload_build_failure_aborts(tmp_path, monkeypatch):
    bs, vsrc, _ = _make_bitstream(tmp_path)
    _seed_models_toml(tmp_path)
    _seed_perfetto_sources(tmp_path)
    mod = load_step("regression/02_eval_performance_event.step.py")
    _patch_paths(mod, monkeypatch, tmp_path, vsrc)

    def fake(**k):
        prefix = k.get("stdout_prefix", "")
        if prefix.startswith("workload build"):
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(mod, "stream_run_logger", fake)
    ctx = FakeCtx()
    run(mod.handler({"chip": "pebble", "bitstream": bs,
                     "_trace_id": "tid-origin"}, ctx))
    failure = ctx.state.store[("tid-origin", "failure")]["body"]
    assert failure["error"] == "workload_build_failed"
    assert failure["model"] == "lenet"
    assert not (tmp_path / "chipcrowd-eval-result.json").exists()
