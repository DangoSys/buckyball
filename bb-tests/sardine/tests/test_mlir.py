import pytest
import logging
import time
import os
from pathlib import Path


sardine_dir = Path(__file__).parent.parent
mlir_toy_workload_dir = (
    sardine_dir / ".." / "output" / "workloads" / "src" / "OpTest" / "toy"
)
mlir_tile_workload_dir = (
    sardine_dir / ".." / "output" / "workloads" / "src" / "OpTest" / "tile"
)


# Define all MLIR OpTest workloads (binary name, test id)
mlir_workloads = [
    ("bb_mvin_mvout_singlecore-baremetal", "bb_mvin_mvout"),
    ("bb_dma1_singlecore-baremetal", "bb_dma1"),
    ("bb_dma2_singlecore-baremetal", "bb_dma2"),
    ("bb_dma3_singlecore-baremetal", "bb_dma3"),
    ("bb_mul_warp16_singlecore-baremetal", "bb_mul_warp16"),
    ("bb_im2col_singlecore-baremetal", "bb_im2col"),
    ("bb_quant_dequant_singlecore-baremetal", "bb_quant_dequant"),
]

# Tile-level tests
mlir_tile_workloads = [
    ("tile_matmul_singlecore-baremetal", "tile_matmul"),
    ("tile_transpose_singlecore-baremetal", "tile_transpose"),
    ("tile_conv2d_singlecore-baremetal", "tile_conv2d"),
]


@pytest.mark.verilator
@pytest.mark.mlir
@pytest.mark.parametrize(
    "workload_path, workload_id, test_index",
    [(path, id, idx) for idx, (path, id) in enumerate(mlir_workloads)],
    ids=[w[1] for w in mlir_workloads],
)
def test_mlir_optest(command_run, caplog, workload_path, workload_id, test_index):
    caplog.set_level(logging.INFO)

    time.sleep(test_index * 20)
    start_time = time.time()
    coverage_flag = " --coverage" if os.environ.get("SARDINE_COVERAGE") else ""
    command = f'bbdev verilator --sim "--binary {workload_path} --batch{coverage_flag}"'
    logging.info(f"Running command: {command}")

    early_exit_pattern = (
        r"Task completed\. Command running on http://localhost:\d+ is finished"
    )
    result = command_run(command, early_exit_pattern=early_exit_pattern, timeout=1200)
    execution_time = time.time() - start_time

    logging.info(f"Workload: {workload_id}")
    logging.info(f"Workload path: {workload_path}")
    logging.info(f"Test index: {test_index}")
    logging.info(f"Execution time: {execution_time:.2f} seconds")
    logging.info(f"Return code: {result['returncode']}")
    logging.info("Script output completed")

    min_execution_time = 5.0
    assert (
        execution_time >= min_execution_time
    ), f"Script executed too quickly: {execution_time:.2f}s < {min_execution_time}s"
    assert result["returncode"] in [
        0,
        1,
    ], f"Script failed with unexpected return code: {result['returncode']}"

    if "PASSED" not in result["stdout"]:
        assert False, f"Script failed: {result['stdout']}"

    logging.info("test completed")


@pytest.mark.verilator
@pytest.mark.mlir
@pytest.mark.parametrize(
    "workload_path, workload_id, test_index",
    [(path, id, idx) for idx, (path, id) in enumerate(mlir_tile_workloads)],
    ids=[w[1] for w in mlir_tile_workloads],
)
def test_mlir_tile_optest(command_run, caplog, workload_path, workload_id, test_index):
    caplog.set_level(logging.INFO)

    time.sleep(test_index * 20)
    start_time = time.time()
    coverage_flag = " --coverage" if os.environ.get("SARDINE_COVERAGE") else ""
    command = f'bbdev verilator --sim "--binary {workload_path} --batch{coverage_flag}"'
    logging.info(f"Running command: {command}")

    early_exit_pattern = (
        r"Task completed\. Command running on http://localhost:\d+ is finished"
    )
    result = command_run(command, early_exit_pattern=early_exit_pattern, timeout=1200)
    execution_time = time.time() - start_time

    logging.info(f"Workload: {workload_id}")
    logging.info(f"Workload path: {workload_path}")
    logging.info(f"Test index: {test_index}")
    logging.info(f"Execution time: {execution_time:.2f} seconds")
    logging.info(f"Return code: {result['returncode']}")
    logging.info("Script output completed")

    min_execution_time = 5.0
    assert (
        execution_time >= min_execution_time
    ), f"Script executed too quickly: {execution_time:.2f}s < {min_execution_time}s"
    assert result["returncode"] in [
        0,
        1,
    ], f"Script failed with unexpected return code: {result['returncode']}"

    if "PASSED" not in result["stdout"]:
        assert False, f"Script failed: {result['stdout']}"

    logging.info("test completed")
