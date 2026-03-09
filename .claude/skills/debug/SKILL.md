---
name: debug
description: 系统性调试 Buckyball 仿真失败。当仿真返回 FAILED、CTest 测试不通过、Chisel 编译报错，或用户报告 Ball 行为异常时使用此 skill。涵盖日志分析、波形分析、常见故障模式匹配。即使用户只是说"仿真失败了"或"跑不过"也应触发。
---

**重要：编译、仿真等操作必须通过 MCP 工具调用，禁止直接使用 bbdev CLI 或 nix develop 命令。**

## 第一步 — 定位日志

1. 找到日志目录：
   - MCP 工具返回的 JSON 中包含 `log_dir` 字段
   - 如果没有，用 `ls -t arch/log/ | head -5` 找最近的日志目录
2. 确认关键日志文件存在：
   - `stdout.log` — 程序标准输出（PASSED/FAILED、printf）
   - `disasm.log` — 反汇编指令流
   - `bdb.log` — Buckyball 硬件调试日志（最重要）
   - `bbdev/server.log` — bbdev server 日志（编译错误在这里）

## 第二步 — 分层分析

按照从高层到底层的顺序分析，先排除简单问题：

### Level 1: 编译错误（bbdev/server.log）

如果 MCP 工具返回 HTTP 500 或 returncode=1，先看 server.log：
- Chisel 编译错误（类型不匹配、缺少注册等）
- mill 构建错误（依赖问题）
- CTest 编译错误（C 语法、链接问题）

### Level 2: 程序输出（stdout.log）

- 搜索 `PASSED` / `FAILED` 确认测试结果
- 搜索 `printf` 输出，检查实际值 vs 预期值
- 搜索 `panic` / `abort` / `trap` 看是否有异常

### Level 3: 指令流（disasm.log）

- 确认 Ball 的 custom 指令确实被执行了（搜索 `custom3`）
- 检查指令顺序是否正确（mvin → ball_op → mvout → fence）
- 检查是否有 trap 或 exception

### Level 4: 硬件跟踪（bdb.log）

这是定位 Ball 逻辑错误最重要的日志，包含三种 trace：

**[ITRACE] 指令 trace：**
- `ISSUE rob_id=X domain=Y funct=0xZZ` — 指令发射
- `COMPLETE rob_id=X` — 指令完成
- 检查：指令是否被发射？是否完成？完成顺序是否正确？

**[MTRACE] 内存 trace：**
- `READ ch=X vbank=Y group=Z addr=0xAA` — SRAM 读
- `WRITE ch=X vbank=Y group=Z addr=0xAA data=0x...` — SRAM 写
- 检查：读写地址是否正确？数据是否正确？bank_id 是否匹配？

**[PMCTRACE] 性能计数 trace：**
- `BALL ball_id=X rob_id=Y elapsed=Z` — Ball 操作耗时
- `LOAD/STORE rob_id=X elapsed=Y` — 内存操作耗时
- 检查：elapsed 是否合理？是否有异常长的操作？

### Level 5: 波形分析（waveform-mcp）

如果日志分析无法定位问题，使用 waveform-mcp 做 cycle-level 分析。详见 `/waveform` skill。

## 第三步 — 常见故障模式

### 1. Ball 没有响应（cmdResp 永远不 fire）
**症状：** 仿真超时或死锁，bdb.log 中有 ISSUE 但没有 COMPLETE
**原因：**
- FSM 卡在某个状态（检查 state 转移条件）
- SRAM resp.valid 没被处理（忘了 resp.ready := true.B）
- cmdResp.valid 没有被拉高

### 2. 数据全零
**症状：** CTest 报 FAILED，输出矩阵全是 0
**原因：**
- 写操作 addr 错误（waddr 没有递增）
- 写操作 mask 全零（忘了设置 mask := 1）
- bank_id 错误（写到了错误的 bank）

### 3. 数据不变（输出 == 输入）
**症状：** CTest 报 FAILED，输出矩阵等于输入矩阵
**原因：**
- 计算逻辑没有生效（跳过了 compute 状态）
- 读到的数据没有被处理就直接写回了

### 4. 部分数据错误
**症状：** CTest 报 FAILED，部分行正确部分行错误
**原因：**
- iter 次数计算错误（少读/少写了几行）
- 地址偏移量计算错误（行之间的 stride）
- 边界条件处理错误

### 5. SRAM 时序错误
**症状：** 数据看起来"偏移了一行"
**原因：**
- SRAM 读延迟是 1 cycle，但代码在 req.fire 的同一个 cycle 就取了 resp.bits.data
- 正确做法：req.fire 后等一个 cycle，在下一个 cycle resp.valid 时读取数据

### 6. bank_id 冲突
**症状：** assertion failure 或数据错乱
**原因：**
- op1_bank 和 wr_bank 使用了同一个 bank（读写冲突）
- 多个 Ball 同时访问同一个 bank

### 7. rob_id 不匹配
**症状：** 指令完成顺序混乱
**原因：**
- cmdResp.bits.rob_id 没有返回正确的 rob_id
- rob_id 没有在 cmdReq.fire 时被 latch

## 第四步 — 修复 + 验证

1. 结合日志/波形分析定位到的问题，修改 Chisel 源码
2. 重新通过 MCP 工具编译和仿真验证修复结果
3. 如果修复引入新问题，回到第二步重新分析
