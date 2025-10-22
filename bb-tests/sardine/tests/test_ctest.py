import pytest
import logging
import time
from pathlib import Path
import subprocess


# ctest_workload_dir = Path("") // absolute path
sardine_dir = Path(__file__).parent.parent
ctest_workload_dir = sardine_dir / ".." / "output" / "workloads" / "src" / "CTest"


# Define all ctest workloads with absolute paths and corresponding IDs
ctest_workloads = [
    (
        "ctest_mvin_mvout_test_singlecore-baremetal",
        "ctest_mvin_mvout_test_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_matmul_16xn_ones_singlecore-baremetal",
        "ctest_vecunit_matmul_16xn_ones_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_matmul_16xn_random1_singlecore-baremetal",
        "ctest_vecunit_matmul_16xn_random1_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_matmul_16xn_random2_singlecore-baremetal",
        "ctest_vecunit_matmul_16xn_random2_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_matmul_16xn_random3_singlecore-baremetal",
        "ctest_vecunit_matmul_16xn_random3_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_matmul_16xn_zero_random_singlecore-baremetal",
        "ctest_vecunit_matmul_16xn_zero_random_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_matmul_col_row_vector_singlecore-baremetal",
        "ctest_vecunit_matmul_col_row_vector_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_matmul_identity_random_singlecore-baremetal",
        "ctest_vecunit_matmul_identity_random_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_matmul_ones_singlecore-baremetal",
        "ctest_vecunit_matmul_ones_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_matmul_random1_singlecore-baremetal",
        "ctest_vecunit_matmul_random1_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_matmul_random2_singlecore-baremetal",
        "ctest_vecunit_matmul_random2_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_matmul_random3_singlecore-baremetal",
        "ctest_vecunit_matmul_random3_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_matmul_zero_random_singlecore-baremetal",
        "ctest_vecunit_matmul_zero_random_singlecore-baremetal",
    ),
    (
        "ctest_vecunit_simple_nn_forward_pass_test_singlecore-baremetal",
        "ctest_vecunit_simple_nn_forward_pass_test_singlecore-baremetal",
    ),
    (
        "ctest_gelu_test_singlecore-baremetal",
        "ctest_gelu_test_singlecore-baremetal",
    ),
    (
        "ctest_softmax_test_singlecore-baremetal",
        "ctest_softmax_test_singlecore-baremetal",
    ),
    (
        "ctest_relu_test_singlecore-baremetal",
        "ctest_relu_test_singlecore-baremetal",
    ),
]


@pytest.mark.verilator
@pytest.mark.ctest
@pytest.mark.parametrize(
    "workload_path, workload_id, test_index",
    [(path, id, idx) for idx, (path, id) in enumerate(ctest_workloads)],
    ids=[w[1] for w in ctest_workloads],
)
def test_ctest_workload_debug(
    command_run, caplog, workload_path, workload_id, test_index
):
    caplog.set_level(logging.INFO)

    time.sleep(test_index * 20)
    start_time = time.time()
    command = f'source {sardine_dir}/../../env.sh && bbdev verilator --sim "--binary {workload_path} --batch"'
    logging.info(f"Running command: {command}")

    # 使用 command_run 执行命令，带提前退出检测
    early_exit_pattern = (
        r"Task completed\. Command running on http://localhost:\d+ is finished"
    )
    result = command_run(
        command, early_exit_pattern=early_exit_pattern, timeout=1200
    )  # 20 minutes
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

    # if '"success":true,"failure":false,' not in result["stdout"]:
    # assert False, f"Script failed: {result['stdout']}"

    if "PASSED" not in result["stdout"]:
        assert False, f"Script failed: {result['stdout']}"

    logging.info("test completed")
