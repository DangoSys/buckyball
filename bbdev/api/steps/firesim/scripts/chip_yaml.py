"""Prepare per-chip FireSim manager YAMLs from chip sims.firesim."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def recipe_name(chip: str) -> str:
    return f"alveo_u280_firesim_{chip}_no_nic"


def _latest_bitstream(results: Path, recipe: str) -> Optional[Path]:
    if not results.is_dir():
        return None
    dirs = sorted(
        (p for p in results.iterdir() if p.is_dir() and recipe in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    for d in reversed(dirs):
        matches = list(d.rglob("firesim.tar.gz"))
        if matches:
            return matches[0]
    return None


def prepare_yamls(bbdir: str, chip: str, target_config: str) -> str:
    """Write chip-scoped firesim yamls under scripts/yaml/<chip>. Return that dir."""
    if not target_config or "." not in target_config:
        raise ValueError(f"invalid firesim TARGET_CONFIG: {target_config!r}")
    recipe = recipe_name(chip)
    script_dir = Path(bbdir) / "bbdev" / "api" / "steps" / "firesim" / "scripts"
    out = script_dir / "yaml" / chip
    out.mkdir(parents=True, exist_ok=True)

    deploy = Path(bbdir) / "thirdparty" / "firesim" / "deploy"
    makefrag = script_dir / "makefrag" / "firesim"
    if not makefrag.is_dir():
        raise FileNotFoundError(f"missing firesim makefrag: {makefrag}")

    build_dir = deploy / "FIRESIM_BUILD_DIR"
    runs_dir = deploy / "FIRESIM_RUNS_DIR"
    build_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    bitstream = _latest_bitstream(deploy / "results-build", recipe)
    if bitstream is None:
        bitstream_line = (
            "    # bitstream_tar filled after buildbitstream; required for infrasetup/runworkload\n"
            "    bitstream_tar: null"
        )
    else:
        bitstream_line = f"    bitstream_tar: file://{bitstream.resolve()}"

    (out / "config_build_recipes.yaml").write_text(
        f"""# Auto-generated for chip={chip}. Do not edit by hand.
{recipe}:
    PLATFORM: xilinx_alveo_u280
    TARGET_PROJECT: firesim
    TARGET_PROJECT_MAKEFRAG: {makefrag}
    DESIGN: FireSim
    TARGET_CONFIG: {target_config}
    PLATFORM_CONFIG: BaseXilinxAlveoU280Config
    deploy_quintuplet: null
    platform_config_args:
        fpga_frequency: 30
        build_strategy: TIMING
    post_build_hook: null
    metasim_customruntimeconfig: null
    bit_builder_recipe: bit-builder-recipes/xilinx_alveo_u280.yaml
""",
        encoding="utf-8",
    )

    (out / "config_build.yaml").write_text(
        f"""# Auto-generated for chip={chip}. Do not edit by hand.
build_farm:
  base_recipe: build-farm-recipes/externally_provisioned.yaml
  recipe_arg_overrides:
    default_build_dir: {build_dir}
    build_farm_hosts:
        - localhost

builds_to_run:
    - {recipe}

agfis_to_share: []

share_with_accounts: {{}}
""",
        encoding="utf-8",
    )

    (out / "config_runtime.yaml").write_text(
        f"""# Auto-generated for chip={chip}. Do not edit by hand.
run_farm:
  base_recipe: run-farm-recipes/externally_provisioned.yaml
  recipe_arg_overrides:
    default_platform: XilinxAlveoU280InstanceDeployManager
    default_simulation_dir: {runs_dir}
    default_fpga_db: /opt/firesim-db.json
    run_farm_hosts_to_use:
        - localhost: one_fpgas_spec

metasimulation:
  metasimulation_enabled: false
  metasimulation_host_simulator: verilator
  metasimulation_only_plusargs: "+fesvr-step-size=128 +max-cycles=100000000"
  metasimulation_only_vcs_plusargs: "+vcs+initreg+0 +vcs+initmem+0"

target_config:
    topology: no_net_config
    no_net_num_nodes: 1
    link_latency: 6405
    switching_latency: 10
    net_bandwidth: 200
    profile_interval: -1
    default_hw_config: {recipe}
    plusarg_passthrough: ""

tracing:
    enable: no
    output_format: 0
    selector: 1
    start: 0
    end: -1

autocounter:
    read_rate: 0

workload:
    workload_name: interactive.json
    terminate_on_completion: no
    suffix_tag: null

host_debug:
    zero_out_dram: no
    disable_synth_asserts: no

synth_print:
    start: 0
    end: -1
    cycle_prefix: yes
""",
        encoding="utf-8",
    )

    (out / "config_hwdb.yaml").write_text(
        f"""# Auto-generated for chip={chip}. bitstream_tar updated by buildbitstream / results-build scan.
{recipe}:
{bitstream_line}
    deploy_quintuplet_override: null
    custom_runtime_config: null
""",
        encoding="utf-8",
    )

    return str(out)


def firesim_cmd(action: str, yaml_dir: str) -> str:
    return (
        f"firesim {action}"
        f" -a {yaml_dir}/config_hwdb.yaml"
        f" -b {yaml_dir}/config_build.yaml"
        f" -r {yaml_dir}/config_build_recipes.yaml"
        f" -c {yaml_dir}/config_runtime.yaml"
    )
