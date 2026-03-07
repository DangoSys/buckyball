验证名为 $ARGUMENTS 的 Ball 的功能正确性。

**重要：编译、仿真、测试等操作必须通过 MCP 工具（bbdev_workload_build、bbdev_verilator_run、bbdev_sardine_run 等）调用，禁止直接使用 bbdev CLI 或 nix develop 命令。**

## 阶段 1 — 完整性检查

检查以下各项是否存在，缺什么补什么：
1. Ball 是否在 `arch/src/main/scala/framework/balldomain/configs/default.json` 中注册
2. ISA 宏是否存在：`bb-tests/workloads/lib/bbhw/isa/` 下对应的 .c 文件
3. CTest 是否存在：`bb-tests/workloads/src/CTest/toy/` 下对应的 _test.c 文件
4. sardine 列表是否包含：`bb-tests/sardine/tests/test_ctest.py` 的 ctest_workloads 列表

## 阶段 2 — 编译 + 仿真

1. 调用 MCP 工具 `bbdev_workload_build` 编译所有 CTest
2. 调用 MCP 工具 `bbdev_verilator_run` 仿真该 Ball 的 CTest
   - binary 名称格式：`ctest_<name>_test_singlecore-baremetal`
   - 设置 batch=true
3. 如果编译或仿真失败，分析错误并修复

## 阶段 3 — 覆盖率分析

1. 调用 MCP 工具 `bbdev_sardine_run`，设置 coverage=true
2. 读取覆盖率报告：`bb-tests/sardine/reports/coverage/html/` 和 `bb-tests/sardine/reports/coverage/annotated/`
3. 分析该 Ball 的 RTL 行覆盖情况
4. 如果覆盖率不足，建议或自动补充测试用例，补充后重新编译和仿真验证

## 阶段 4 — 失败分析（如有）

如果仿真结果为 FAILED：

1. **找到日志目录**：MCP 工具返回的 JSON 中包含 `log_dir` 字段，如 `arch/log/2026-03-07-12-00-ctest_xxx/`
   - 如果 MCP 返回中没有 `log_dir`，用 `ls -t arch/log/ | head -5` 找最近的日志目录

2. **读取关键日志**：
   - `arch/log/<timestamp>/stdout.log` — 程序标准输出，包含 PASSED/FAILED 和 printf 输出
   - `arch/log/<timestamp>/disasm.log` — 反汇编指令流，可以看到实际执行了哪些指令
   - `arch/log/<timestamp>/bdb.log` — **Buckyball 调试日志**，包含 Ball 内部状态变化、SRAM 读写请求/响应、FSM 状态转移等硬件级信息。这是定位 Ball 逻辑错误最重要的日志
   - `bbdev/server.log` — bbdev server 的运行日志，包含 Chisel 编译错误、mill 构建错误等详细堆栈信息

3. **分析 bdb.log 的方法**：
   - 搜索目标 Ball 的名称，找到相关的命令发射和完成事件
   - 检查 SRAM 读写地址和数据是否正确
   - 检查 bank_id 分配是否和 CTest 中 bb_mem_alloc 一致
   - 检查迭代次数（iter）是否正确

4. 结合日志分析 Ball 的 Chisel 源码，定位时序或逻辑问题
5. 提出修复方案并实施
6. 重新编译和仿真验证修复结果（通过 MCP 工具）
