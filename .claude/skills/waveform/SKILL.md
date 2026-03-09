---
name: waveform
description: 使用 waveform-mcp 工具分析 VCD/FST 波形文件，检查 Buckyball 模块的信号时序。当需要做 cycle-level 的时序分析、调试握手协议、检查 FSM 状态转移，或用户要求"看波形"、"分析信号"时使用此 skill。也适用于 `/debug` 和 `/optimize` 中需要波形分析的场景。
---

## 工具概览

项目配置了 `waveform-mcp` MCP server，提供以下工具：
- `open_waveform(file_path)` — 打开 VCD/FST 文件
- `list_signals(waveform_id, hierarchy_prefix?, name_pattern?, recursive?)` — 列出信号
- `read_signal(waveform_id, signal_path, time_index|time_indices)` — 读取信号值
- `get_signal_info(waveform_id, signal_path)` — 获取信号元数据
- `find_signal_events(waveform_id, signal_path, start?, end?, limit?)` — 找信号变化
- `find_conditional_events(waveform_id, condition, start?, end?, limit?)` — 条件搜索
- `close_waveform(waveform_id)` — 关闭波形

## 波形文件位置

仿真生成的波形文件通常在：
- `arch/log/<timestamp>/` 目录下
- 文件格式：`.vcd` 或 `.fst`
- 用 `find arch/log -name "*.vcd" -o -name "*.fst" | head -5` 查找

## 信号层次结构

Buckyball 的信号层次大致为：
```
TOP
├── BuckyballToy (顶层)
│   ├── bbtile (BBTile)
│   │   ├── buckyball (Buckyball 主体)
│   │   │   ├── frontend
│   │   │   │   ├── globalDecoder
│   │   │   │   └── globalROB
│   │   │   ├── ballDomain
│   │   │   │   └── bbus (BBus)
│   │   │   │       ├── cmdRouter
│   │   │   │       ├── pmc (BallCyclePMC)
│   │   │   │       ├── balls_0 (第一个 Ball，按 ballId 排序)
│   │   │   │       ├── balls_1
│   │   │   │       └── ...
│   │   │   └── memDomain
│   │   │       ├── memFrontend
│   │   │       └── memBackend
```

### 定位 Ball 信号

要找到特定 Ball 的信号：
1. `list_signals(waveform_id, recursive=false)` — 先看顶层
2. 逐级向下：`list_signals(waveform_id, hierarchy_prefix="TOP.BuckyballToy")` 等
3. 或直接用 `name_pattern` 搜索：`list_signals(waveform_id, name_pattern="relu", recursive=true)`

## 常用检查模板

### 1. 握手检查（Decoupled valid/ready）

检查 cmdReq 握手是否成功：
```
find_conditional_events(waveform_id,
  condition="TOP...cmdReq_valid && TOP...cmdReq_ready")
```

检查 cmdResp 握手：
```
find_conditional_events(waveform_id,
  condition="TOP...cmdResp_valid && TOP...cmdResp_ready")
```

### 2. SRAM 读时序检查

验证 SRAM 1-cycle 读延迟：
```
# 找到读请求
find_conditional_events(waveform_id,
  condition="TOP...bankRead_0_io_req_valid && TOP...bankRead_0_io_req_ready")

# 检查下一个 cycle 是否有 resp.valid
# 取上面事件的 time_index + 1，然后
read_signal(waveform_id, "TOP...bankRead_0_io_resp_valid", time_index=<req_time+1>)
```

### 3. FSM 状态转移追踪

找 state 寄存器的所有变化：
```
find_signal_events(waveform_id, signal_path="TOP...state")
```

然后读取每个变化点的值来重建状态转移序列。

### 4. Ball 操作延迟测量

测量从 cmdReq.fire 到 cmdResp.fire 的 cycle 数：
```
# 找 cmdReq 握手时刻
req_events = find_conditional_events(waveform_id,
  condition="TOP...cmdReq_valid && TOP...cmdReq_ready")

# 找 cmdResp 握手时刻
resp_events = find_conditional_events(waveform_id,
  condition="TOP...cmdResp_valid && TOP...cmdResp_ready")

# 对应的 resp - req 即为延迟
```

### 5. Bank 地址/数据检查

在某个时刻读取 SRAM 读写的地址和数据：
```
read_signal(waveform_id, "TOP...bankRead_0_io_req_bits_addr", time_index=T)
read_signal(waveform_id, "TOP...bankRead_0_io_resp_bits_data", time_index=T+1)
read_signal(waveform_id, "TOP...bankWrite_0_io_req_bits_addr", time_index=T)
read_signal(waveform_id, "TOP...bankWrite_0_io_req_bits_data", time_index=T)
```

## 条件搜索语法

`find_conditional_events` 的 condition 支持：
- 信号路径：`TOP.module.signal`
- 位运算：`~`(NOT), `&`(AND), `|`(OR), `^`(XOR)
- 布尔运算：`&&`, `||`, `!`
- 比较：`==`, `!=`
- 位提取：`signal[bit]` 或 `signal[msb:lsb]`
- 前值：`$past(signal)` — 前一个 time_index 的值
- Verilog 字面量：`4'b0001`, `8'hFF`

常用 pattern：
- 上升沿：`!$past(TOP.signal) && TOP.signal`
- 下降沿：`$past(TOP.signal) && !TOP.signal`
- 握手：`TOP.valid && TOP.ready`
- 某位为 1：`TOP.flags & 4'b0001`

## 使用建议

1. **先 list_signals 再 read** — 信号路径名可能和 Chisel 源码中的名称略有不同（Chisel 会做 name mangling），先搜索确认
2. **用 find_conditional_events 而不是逐 cycle read** — 效率差几个数量级
3. **限制 limit** — 波形可能很长，设置合理的 limit 避免返回过多数据
4. **用完记得 close_waveform** — 释放内存
