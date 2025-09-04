import pytest
import logging
import time
from pathlib import Path
import subprocess



# ctest_workload_dir = Path("") // absolute path
sardine_dir = Path(__file__).parent.parent
ctest_workload_dir = sardine_dir / "workloads" / "CTest"


# Define all ctest workloads with absolute paths and corresponding IDs
ctest_workloads = [
  (f"{ctest_workload_dir}/ctest_mvin_mvout_acc_test_singlecore-baremetal", "ctest_mvin_mvout_acc_test_singlecore-baremetal"),
  (f"{ctest_workload_dir}/ctest_mvin_mvout_alternate_test_singlecore-baremetal", "ctest_mvin_mvout_alternate_test_singlecore-baremetal"),
  (f"{ctest_workload_dir}/ctest_vecunit_matmul_16xn_ones_singlecore-baremetal", "ctest_vecunit_matmul_16xn_ones_singlecore-baremetal"),     
  (f"{ctest_workload_dir}/ctest_vecunit_matmul_16xn_random1_singlecore-baremetal", "ctest_vecunit_matmul_16xn_random1_singlecore-baremetal"),
  (f"{ctest_workload_dir}/ctest_vecunit_matmul_16xn_random2_singlecore-baremetal", "ctest_vecunit_matmul_16xn_random2_singlecore-baremetal"),
  (f"{ctest_workload_dir}/ctest_vecunit_matmul_16xn_random3_singlecore-baremetal", "ctest_vecunit_matmul_16xn_random3_singlecore-baremetal"),
  (f"{ctest_workload_dir}/ctest_vecunit_matmul_16xn_zero_random_singlecore-baremetal", "ctest_vecunit_matmul_16xn_zero_random_singlecore-baremetal"),
  (f"{ctest_workload_dir}/ctest_vecunit_matmul_col_row_vector_singlecore-baremetal", "ctest_vecunit_matmul_col_row_vector_singlecore-baremetal"),   
  (f"{ctest_workload_dir}/ctest_vecunit_matmul_identity_random_singlecore-baremetal", "ctest_vecunit_matmul_identity_random_singlecore-baremetal"),
  (f"{ctest_workload_dir}/ctest_vecunit_matmul_ones_singlecore-baremetal", "ctest_vecunit_matmul_ones_singlecore-baremetal"),
  (f"{ctest_workload_dir}/ctest_vecunit_matmul_random1_singlecore-baremetal", "ctest_vecunit_matmul_random1_singlecore-baremetal"),
  (f"{ctest_workload_dir}/ctest_vecunit_matmul_random2_singlecore-baremetal", "ctest_vecunit_matmul_random2_singlecore-baremetal"),
  (f"{ctest_workload_dir}/ctest_vecunit_matmul_random3_singlecore-baremetal", "ctest_vecunit_matmul_random3_singlecore-baremetal"), 
  (f"{ctest_workload_dir}/ctest_vecunit_matmul_zero_random_singlecore-baremetal", "ctest_vecunit_matmul_zero_random_singlecore-baremetal"),
  (f"{ctest_workload_dir}/ctest_vecunit_simple_nn_forward_pass_test_singlecore-baremetal", "ctest_vecunit_simple_nn_forward_pass_test_singlecore-baremetal"),
]

# @pytest.fixture(scope="session", autouse=True)
# def setup_before_all_tests():
#   """在所有测试开始前执行一次的操作"""
#   logging.info("=== 开始执行 CTest 测试套件 ===")
#   logging.info(f"测试目录: {sardine_dir}")
#   logging.info(f"CTest 工作负载目录: {ctest_workload_dir}")
#   logging.info(f"工作负载数量: {len(ctest_workloads)}")
  
#   # 检查工作负载目录是否存在
#   if not ctest_workload_dir.exists():
#     raise FileNotFoundError(f"CTest 工作负载目录不存在: {ctest_workload_dir}")
  
#   # 检查每个工作负载是否存在
#   for workload_path, workload_id in ctest_workloads:
#     if not Path(workload_path).exists():
#       logging.warning(f"工作负载不存在: {workload_path} ({workload_id})")
  
#   # 启动所有需要的 bbdev 服务
#   processes = []
#   logging.info("=== 启动所有 bbdev 服务 ===")
#   for idx, (workload_path, workload_id) in enumerate(ctest_workloads):
#     port = base_port + idx
#     start_cmd = f"source {sardine_dir}/../../env.sh && bbdev start --port {port}"
#     logging.info(f"启动服务 {idx}: {start_cmd}")
    
#     process = subprocess.Popen(start_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     processes.append((process, workload_id))
  
#   # 等待所有服务启动
#   logging.info("等待所有服务启动...")
#   time.sleep(30)
  
#   logging.info("=== 测试套件初始化完成 ===\n")

#   yield  # 这里可以添加测试结束后的清理操作
  
#   # 清理所有服务
#   logging.info("=== 清理所有 bbdev 服务 ===")
#   for process, workload_id in processes:
#     logging.info(f"停止服务 {workload_id}")
#     if process.poll() is None:
#       process.terminate()
#       try:
#         process.wait(timeout=10)
#       except subprocess.TimeoutExpired:
#         process.kill()
#         process.wait()
  
#   # 统一停止所有 bbdev 实例
#   # stop_cmd = f"source {sardine_dir}/../../env.sh && bbdev stop"
#   # logging.info(f"执行统一停止命令: {stop_cmd}")
#   # subprocess.run(stop_cmd, shell=True, timeout=30)
  
#   logging.info("=== CTest 测试套件执行完成 ===")


# @pytest.fixture(scope="function", autouse=True)
# def setup_before_each_test(script_runner, request):
#   """在每个测试开始前执行的操作"""
#   logging.info("--- 开始执行单个测试 ---")

#   yield
  
#   logging.info("--- 单个测试执行完成 ---\n")


@pytest.mark.verilator
@pytest.mark.ctest
@pytest.mark.parametrize("workload_path, workload_id, test_index", 
                         [(path, id, idx) for idx, (path, id) in enumerate(ctest_workloads)], 
                         ids=[w[1] for w in ctest_workloads])
def test_ctest_workload_debug(command_run, caplog, workload_path, workload_id, test_index):
  caplog.set_level(logging.INFO)
  
  time.sleep(test_index * 10)
  start_time = time.time()
  command = f"source {sardine_dir}/../../env.sh && bbdev verilator --sim \"--binary {workload_path} --batch\""
  logging.info(f"Running command: {command}")
  
  # 使用 command_run 执行命令，带提前退出检测
  early_exit_pattern = r'Task completed\. Command running on http://localhost:\d+ is finished'
  result = command_run(command, early_exit_pattern=early_exit_pattern, timeout=3600) # 60 minutes
  execution_time = time.time() - start_time
  
  logging.info(f"Workload: {workload_id}")
  logging.info(f"Workload path: {workload_path}")
  logging.info(f"Test index: {test_index}")
  logging.info(f"Execution time: {execution_time:.2f} seconds")
  logging.info(f"Return code: {result['returncode']}")
  logging.info("Script output completed")

  min_execution_time = 5.0
  assert execution_time >= min_execution_time, f"Script executed too quickly: {execution_time:.2f}s < {min_execution_time}s"
  assert result['returncode'] in [0, 1], f"Script failed with unexpected return code: {result['returncode']}"

  if f'"success":true,"failure":false,' not in result['stdout']:
    assert False, f"Script failed: {result['stdout']}"

  logging.info("test completed")
