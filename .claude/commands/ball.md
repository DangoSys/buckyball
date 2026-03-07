创建一个名为 $ARGUMENTS 的新 Buckyball Ball 算子，完成从实现到验证的全流程。

**重要：编译、仿真等操作必须通过 MCP 工具（validate、bbdev_workload_build、bbdev_verilator_run 等）调用，禁止直接使用 bbdev CLI 或 nix develop 命令。**

## 阶段 1 — 需求收集

1. 读取当前注册状态，确定新 Ball 的 ballId 和 funct7：
   - `arch/src/main/scala/framework/balldomain/configs/default.json`
   - `arch/src/main/scala/examples/toy/balldomain/DISA.scala`
2. 向用户确认以下信息：
   - Ball 的计算语义（做什么运算）
   - inBW / outBW（读/写 bank 端口数量）
   - 是否需要第二个操作数（op2）
   - iter（迭代次数）的含义

## 阶段 2 — 实现 Ball

1. 读取参考代码，理解 Blink 协议和现有 Ball 的写法：
   - 简单参考：`arch/src/main/scala/framework/balldomain/prototype/relu/ReluBall.scala` 和 `Relu.scala`
   - 复杂参考：`arch/src/main/scala/framework/balldomain/prototype/systolicarray/`
   - Blink 协议：`arch/src/main/scala/framework/balldomain/blink/blink.scala`、`bank.scala`、`status.scala`
   - SRAM 接口：`arch/src/main/scala/framework/memdomain/backend/banks/SramIO.scala`
2. 在 `arch/src/main/scala/framework/balldomain/prototype/<name>/` 下创建：
   - `<Name>Ball.scala` — wrapper，extends Module with HasBlink，从 ballIdMappings 取 inBW/outBW，实例化 core，连接 BlinkIO
   - `<Name>.scala` — core 计算逻辑，包含 FSM 和数据通路
   - `configs/<Name>BallParam.scala` — 算子参数 case class
   - `configs/default.json` — 算子专属配置

关键约束：
- SRAM 读延迟 = 1 cycle（req.fire 后下一周期 resp.valid）
- cmdReq.fire 时 latch 命令字段到寄存器
- FSM 基本模式：idle → 读数据 → 计算 → 写数据 → complete → idle
- status.idle 和 status.running 必须映射 FSM 状态

## 阶段 3 — 注册

按顺序更新以下四个文件：
1. `arch/src/main/scala/framework/balldomain/configs/default.json` — 追加 ballIdMappings 条目，更新 ballNum
2. `arch/src/main/scala/examples/toy/balldomain/bbus/busRegister.scala` — 添加 import 和 match case
3. `arch/src/main/scala/examples/toy/balldomain/DISA.scala` — 添加 `val XXX = BitPat("bxxxxxxx")`
4. `arch/src/main/scala/examples/toy/balldomain/DomainDecoder.scala` — 添加 ListLookup 解码行，BID = ballId.U

## 阶段 4 — ISA C 宏

1. 在 `bb-tests/workloads/lib/bbhw/isa/` 下创建 `<funct7十进制>_<name>.c`
   - 定义 `BB_<NAME>_FUNC7` 宏（funct7 十进制值）
   - 定义 `bb_<name>(...)` 宏（编码 rs1/rs2 + 调用 BUCKYBALL_INSTRUCTION_R_R）
   - 参考：`bb-tests/workloads/lib/bbhw/isa/38_relu.c`
2. 在 `bb-tests/workloads/lib/bbhw/isa/isa.h` 中 `#include` 新文件

## 阶段 5 — CTest

1. 在 `bb-tests/workloads/src/CTest/toy/` 下创建 `<name>_test.c`
   - 包含固定输入矩阵和预期输出矩阵
   - 遵循 bb_mem_alloc → bb_mvin → bb_<name> → bb_mvout → bb_fence 流程
   - compare 判定 PASS/FAIL
   - 参考：`bb-tests/workloads/src/CTest/toy/relu_test.c`
2. 在 `bb-tests/workloads/src/CTest/toy/CMakeLists.txt` 中用 `add_cross_platform_test_target` 注册
3. 在 `bb-tests/sardine/tests/test_ctest.py` 的 `ctest_workloads` 列表中追加对应条目

## 阶段 6 — 校验 + 编译 + 仿真

1. 调用 MCP 工具 `validate` 做静态校验，确认 6 项不变量全部通过
2. 调用 MCP 工具 `bbdev_workload_build` 编译 CTest
3. 调用 MCP 工具 `bbdev_verilator_run` 跑仿真，指定新 Ball 的 CTest binary
4. 解析仿真结果：
   - PASSED → 完成
   - FAILED → 进入调试：
     a. 找到日志目录（MCP 返回的 `log_dir`，或 `ls -t arch/log/ | head -5`）
     b. 读取 `stdout.log`（程序输出）、`disasm.log`（反汇编）、**`bdb.log`（Ball 硬件调试日志，最重要）**
     c. 在 bdb.log 中搜索 Ball 名称，检查 SRAM 读写地址/数据、bank_id 分配、iter 次数
     d. 结合 Chisel 源码定位问题并修复
     e. 重新通过 MCP 工具编译和仿真验证
