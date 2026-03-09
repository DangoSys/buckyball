---
name: verify
description: 验证名为 $ARGUMENTS 的 Ball 的功能正确性。当用户要求验证、测试某个 Ball，检查 Ball 是否工作正常，或说"验证 XX Ball"、"跑一下 XX 测试"时使用此 skill。也适用于用户新建完 Ball 后想确认其正确性的场景。
---

**重要：编译、仿真、测试等操作必须通过 MCP 工具调用，禁止直接使用 bbdev CLI 或 nix develop 命令。**

## 阶段 1 — 完整性检查

使用 `/check` 的逻辑检查注册一致性，然后检查以下各项是否存在，缺什么补什么：
1. Ball 实现：`arch/src/main/scala/framework/balldomain/prototype/<name>/` 目录是否存在
2. 注册：`arch/src/main/scala/framework/balldomain/configs/default.json` 中是否有条目
3. ISA 宏：`bb-tests/workloads/lib/bbhw/isa/` 下对应的 .c 文件
4. CTest：`bb-tests/workloads/src/CTest/toy/` 下对应的 _test.c 文件
5. sardine 列表：`bb-tests/sardine/tests/test_ctest.py` 的 ctest_workloads 列表

## 阶段 2 — 编译 + 仿真

1. 调用 MCP 工具 `bbdev_workload_build` 编译所有 CTest
2. 调用 MCP 工具 `bbdev_verilator_run` 仿真该 Ball 的 CTest
   - binary 名称格式：`ctest_<name>_test_singlecore-baremetal`
   - 设置 batch=true
3. 如果编译或仿真失败，使用 `/debug` skill 进入调试流程

## 阶段 3 — 覆盖率分析

1. 调用 MCP 工具 `bbdev_sardine_run`，设置 coverage=true
2. 读取覆盖率报告：
   - 行覆盖数据在 `bb-tests/sardine/reports/coverage/annotated/` 下
   - 找到对应 Ball 的文件：grep 搜索 Ball 类名
3. 分析该 Ball 的 RTL 行覆盖情况：
   - 查看未覆盖的行（标记为 `000000`）
   - 重点关注 FSM 状态、边界条件、错误路径
4. 如果覆盖率不足，建议或自动补充测试用例，补充后重新编译和仿真验证

## 阶段 4 — PMC 性能分析

仿真通过后，读取 bdb.log 中的 PMC trace 数据来分析性能：

1. 找到日志目录（`ls -t arch/log/ | head -5`）
2. 在 bdb.log 中搜索 `[PMCTRACE] BALL` 条目，提取该 Ball 的 elapsed cycle 数据
3. 汇总报告：
   - 平均 elapsed cycles per task
   - 最大/最小 elapsed cycles
   - 总调用次数

## 阶段 5 — 波形分析（仿真失败时）

如果仿真失败，除了读日志外，还可以用 waveform-mcp 做精确的时序分析。详见 `/waveform` skill。

关键信号检查清单：
- `cmdReq.valid && cmdReq.ready`（命令握手）
- SRAM `req.valid/ready` 和 `resp.valid`（读写时序）
- FSM state 寄存器（状态转移）
- `cmdResp.valid && cmdResp.fire`（完成握手）

## 失败分析

如果仿真结果为 FAILED，使用 `/debug` skill 进行系统性调试。
