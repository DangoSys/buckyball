分析并优化名为 $ARGUMENTS 的 Ball 的延迟和面积。

**重要：综合、仿真等操作必须通过 MCP 工具（bbdev_yosys_synth、bbdev_verilator_run、bbdev_workload_build 等）调用，禁止直接使用 bbdev CLI 或 nix develop 命令。**

## 阶段 1 — 基线测量

1. 调用 MCP 工具 `bbdev_yosys_synth` 跑 Yosys 综合 + OpenSTA 时序分析
2. 读取面积报告：`bbdev/api/steps/yosys/log/hierarchy_report.txt`（子模块分解）
3. 读取时序报告：`bbdev/api/steps/yosys/log/timing_report.txt`（关键路径）
4. 检查该 Ball 的 CTest 中是否有 `read_rdcycle()` 计时代码
   - 如果有：调用 `bbdev_verilator_run` 跑仿真获取基线 cycle 数
   - 如果没有：自动在 CTest 中加入 read_rdcycle() 前后对比代码，调用 `bbdev_workload_build` 重新编译，再调用 `bbdev_verilator_run` 获取基线 cycle 数

## 阶段 2 — 面积分析

从 hierarchy_report.txt 提取目标 Ball 及其子模块的面积数据：
- 总面积（Chip area）
- Sequential 占比（寄存器面积）
- Combinational 占比（组合逻辑面积）
- 子模块面积排名

识别面积大户：
- Sequential 占比高 → 寄存器多，考虑是否可用 SRAM 替代
- Combinational 占比高 → 逻辑复杂，考虑是否可以简化或共享

对比同类 Ball 面积（如 ReluBall vs TransposeBall），找出效率差距。

## 阶段 3 — 时序/延迟分析

1. 从 timing_report.txt 看关键路径是否经过该 Ball，以及路径延迟
2. 从仿真结果看操作延迟（cycle 数）
3. 读取 Ball 的 Chisel core 源码，分析 FSM：
   - 绘制 FSM 状态转移图
   - 计算每个状态的 cycle 数（最佳/最坏情况）
   - 识别瓶颈状态（哪个状态耗时最多）
   - 分析 SRAM 读写模式（串行 vs 流水 vs 多端口并行）

## 阶段 4 — 优化方案

提出可量化的优化方案，每个方案包含：
- 优化手段描述
- 预期面积变化（参考 hierarchy_report 数据量化）
- 预期延迟变化（cycle 数）
- 预期频率影响
- trade-off 说明

常见优化模式：

**降延迟**：
- 读写流水化：让写操作和下一轮读操作重叠，面积略增，延迟显著降
- 多 bank 端口并行读：利用 inBW > 1 同时读多行，面积不变，延迟与端口数成比例降
- 去中间等待状态：合并不必要的 FSM 状态，面积略降
- 边读边算：计算嵌入读响应周期，利用 SRAM 1-cycle 延迟的下降沿做计算

**降面积**：
- regArray 改 SRAM：大块寄存器阵列换成 SRAM 端口访问，面积显著降，可能增延迟
- 共享计算单元：多操作时分复用同一计算单元，面积降，延迟可能增
- 减少 counter 位宽：根据实际范围缩减位宽，面积微降

**提频率**：
- 拆分组合逻辑长路径：在关键路径中间插入寄存器，面积略增，可能增 1 cycle 延迟

列出方案后让用户选择。

## 阶段 5 — 实施

根据用户选择的方案修改 Chisel 代码。

## 阶段 6 — 优化后测量

1. 再次调用 `bbdev_yosys_synth`，对比 hierarchy_report
2. 再次调用 `bbdev_verilator_run`，对比 cycle 数
3. 调用 `validate` 确认注册一致性未被破坏
4. 输出优化前后对比报告：
   - 面积差值和百分比变化
   - Cycle 数差值和百分比变化
   - 关键路径延迟变化
   - 频率变化

## 故障排查

如果 MCP 工具返回 HTTP 500 或 returncode=1：
- 读取 `bbdev/server.log` — bbdev server 的运行日志，包含 Chisel 编译错误、mill 构建错误等详细堆栈信息
- 该日志特别有助于诊断 elaborate 失败（如类型不匹配、缺少注册等编译期错误）
