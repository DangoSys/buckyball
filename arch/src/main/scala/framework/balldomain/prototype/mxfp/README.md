# MXFP 格式转换加速器

## Overview

本目录实现了 Buckyball 的 **MXFP（Mixed Floating-Point）格式转换加速器**，位置位于 `arch/src/main/scala/framework/balldomain/prototype/mxfp`。
该模块用于从 Scratchpad 中读取 FP32 数据块，按照当前定义的 MXFP 打包格式完成转换，并将结果写回目标 Scratchpad bank。

当前实现重点验证以下完整链路：

- 自定义 MXFP 指令接入
- Ball domain 指令解码与调度
- Scratchpad 读写数据流
- RTL 打包逻辑
- C 端软件黄金模型对比
- Verilator 仿真验证

当前版本属于 **V1 原型实现**，目标是先完成 **FP32 block -> MXFP packed block** 的功能验证。

核心组件：

- **Mxfp.scala**：MXFP 主执行逻辑
- **MxfpBall.scala**：MXFP Ball 外层封装

## Code Structure

```text
mxfp/
├── Mxfp.scala                 - MXFP 主执行逻辑
├── MxfpBall.scala             - MXFP Ball 外层封装
└── configs/
    ├── MxfpBallParam.scala    - 参数定义与读取
    └── default.json           - 默认参数配置
````

### Module Responsibilities

**Mxfp.scala**（加速器实现层）

* 从 Scratchpad 中按 block 读取 FP32 数据
* 收集一个 block 所需的 16 个 FP32 元素
* 计算 global exponent 与 micro bits
* 执行 4-bit magnitude 量化
* 生成 16 个 payload，并打包为一个 128-bit MXFP block
* 将结果写回目标 Scratchpad bank
* 提供 Ball domain 命令接口，并返回完成响应/状态

**MxfpBall.scala**（Ball 封装层）

* 实例化 `PipelinedMxfp`
* 连接 blink 接口
* 透传 bank read / bank write 端口
* 输出状态信息

## Module Description

### Mxfp.scala

**Main functionality**：

当前模块按 block 进行处理：

**从 Scratchpad 连续读取 4 个输入 word -> 拼成 16 个 FP32 元素 -> 执行 MXFP 打包 -> 写回 1 个 128-bit 输出 word**

当前 V1 的固定假设如下：

* `InputNum = 16`
* `inputWidth = 32`
* `bankWidth = 128`
* 一个 bank word 含 `4` 个 FP32
* 一个 MXFP block 含 `16` 个 FP32
* 一个 block 对应：

  * `4` 次 bank read
  * `1` 次 bank write

### State machine definition

```scala
val idle :: sRead :: sPack :: sWrite :: complete :: Nil = Enum(5)
val state = RegInit(idle)
```

状态说明：

* **idle**：等待命令
* **sRead**：连续读取一个 block 所需的 4 个输入 word
* **sPack**：将收集到的 16 个 FP32 打包为一个 MXFP block
* **sWrite**：将 packed block 写回目标 bank
* **complete**：返回完成响应，进入下一次处理或回到 idle

### Key registers

```scala
// 输入缓存：保存一个 block 的 16 个 FP32
val fp32Buf = RegInit(VecInit(Seq.fill(InputNum)(0.U(inputWidth.W))))

// 当前 block 的 packed 输出
val packedBlockReg = RegInit(0.U(bankWidth.W))

// 读请求/响应计数器
val readReqCounter  = RegInit(0.U(...))
val readRespCounter = RegInit(0.U(...))

// 地址与控制寄存器
val raddr_reg       = RegInit(0.U(...))
val waddr_reg       = RegInit(0.U(...))
val rbank_reg       = RegInit(0.U(...))
val wbank_reg       = RegInit(0.U(...))
val remainingBlocks = RegInit(0.U(...))
```

### Command parsing

命令进入时：

* 记录 `rob_id / sub_rob_id`
* 记录源 bank 与目标 bank
* 初始化读写地址
* 将 `iter` 解释为 **待处理 block 数**

当前实现中：

* `op1_bank`：输入源 bank
* `wr_bank`：输出目标 bank
* `iter`：block 数量，不再表示“行数”

### Data conversion logic (MXFP)

当前 V1 实现为固定 **MX6-like 打包格式**。

#### 1. Global exponent

对 block 中 16 个 FP32 元素：

* 提取 exponent
* 忽略 zero / subnormal / special 在正常 exponent 竞争中的影响
* 取最大 exponent 作为 `globalExp`

#### 2. Micro bit

每 2 个元素构成一个 pair：

* 若该 pair 的最大 exponent 比 `globalExp` 至少小 1
* 则该 pair 的 micro bit 置 `1`
* 否则置 `0`

如果 micro bit = `1`，该 pair 使用：

* `localExp = globalExp - 1`

否则使用：

* `localExp = globalExp`

#### 3. Payload

每个元素生成一个 5-bit payload：

* `1 bit sign`
* `4 bit magnitude`

即：

```text
payload = sign[1] ++ mag[4]
```

#### 4. Packed layout

当前输出 block 的布局为：

* byte 0：global exponent
* byte 1：8 个 micro bit
* byte 2 ~ byte 11：16 个 payload 打包后的 80 bit
* byte 12 ~ byte 15：0 填充

总计：

* `8 + 8 + 80 = 96 bit`
* 再补零到 `128 bit`

### Scratchpad interface

当前通过 Ball domain 的 bank read / bank write 接口完成数据交换：

* 输入：从源 bank 连续读取 4 个 word
* 输出：向目标 bank 写回 1 个 packed word

每个 block 的地址步进关系为：

* 读地址每次 `+1`
* 累积 4 次读后完成一个 block
* 写地址每个 block `+1`

### Processing flow

1. **idle**

   * 等待 `cmdReq`
   * 解析源 bank、目标 bank、iter
   * 初始化 block 处理计数器

2. **sRead**

   * 连续发起 4 次 bank read
   * 每次读取 1 个 128-bit word
   * 拆出 4 个 FP32 填入 `fp32Buf`

3. **sPack**

   * 当 16 个 FP32 收集完成后
   * 计算 global exponent、micro bits、payload
   * 生成 1 个 packed MXFP block

4. **sWrite**

   * 向目标 bank 写回当前 block 的 packed 结果

5. **complete**

   * 若还有剩余 block，则继续下一轮
   * 否则发出 `cmdResp`，回到 `idle`

## ISA Structure

该模块对应一条 Ball 指令，用于执行：

**从源 Scratchpad bank 读取 FP32 block，转换为 MXFP packed block，并写回目标 bank**

### Function

执行 Scratchpad 数据的块级 MXFP 格式转换。

### func7

```text
55
```

### Instruction

```c
bb_mxfp(op1_bank_id, wr_bank_id, iter)
```

### Parameters

* `op1_bank_id`：输入数据所在 bank
* `wr_bank_id`：输出数据写回 bank
* `iter`：待处理 block 数量

当前实现中：

* 1 个 block = 16 个 FP32
* 1 个 block 需要 4 个输入 word
* 1 个 block 产生 1 个输出 word

## Usage

### Basic flow

1. 将 FP32 输入数据按原始 IEEE754 bit pattern 准备好
2. 使用 `bb_mvin` 将输入搬入源 bank
3. 调用 `bb_mxfp(...)`
4. 使用 `bb_mvout` 将 packed 输出搬出
5. 与软件黄金模型比较结果

### Example workflow

```text
Input FP32 bit patterns
-> bb_mvin
-> bb_mxfp
-> bb_mvout
-> software golden compare
```

## Test Strategy

当前提供了 C 端测试程序 `mxfp_test.c`，用于验证：

* Scratchpad 输入是否正确搬入
* MXFP Ball 是否正确执行 block 打包
* Scratchpad 输出是否与软件黄金模型一致

### Current validation method

当前测试使用：

* 固定 16 元素 block 输入
* 原始 IEEE754 bit pattern 构造输入
* 软件端复现 RTL 当前的打包逻辑
* 与硬件输出逐字节比较

为了避免 baremetal 环境下浮点执行带来的不确定性，测试输入采用 **IEEE754 的原始 32-bit bit pattern** 表示，而不是直接使用 `float` 运算。

## Current Validation Result

当前版本已经完成以下验证：

* MXFP 指令链路接入成功
* RTL elaboration 成功
* Verilator 仿真成功运行
* Scratchpad 输入 -> MXFP 转换 -> Scratchpad 输出 链路打通
* 软件黄金模型与硬件输出一致

在当前测试配置下：

* 已验证前缀输出与软件黄金模型一致
* 已验证一个完整 128-bit 输出 block 与软件黄金模型一致
* 当前仿真测试返回 `PASSED`

这表明当前版本的：

* global exponent 打包
* micro bit 生成
* payload 打包
* bank 读写流程

在已验证测试中是一致的。

## Notes

1. **当前为原型版本**
   这是一个 V1 原型，主要目标是验证数据路径与打包逻辑，不是最终完整 MX 浮点实现。

2. **当前输入固定为 FP32**
   输入元素宽度固定为 32 bit，输入数据以 IEEE754 原始 bit pattern 形式参与测试。

3. **当前 block 大小固定**
   当前仅支持 `16` 元素 block。

4. **当前 bankWidth 固定为 128 bit**
   因此一个 block 需要通过 4 次读取完成收集。

5. **特殊值处理采用简化策略**

   * zero -> 视为 0
   * subnormal -> 视为 0
   * Inf / NaN -> 饱和到最大 magnitude

6. **当前实现重点是格式转换**
   当前不涉及更复杂的 MX 算术计算，仅实现 FP32 -> MXFP packed format。

7. **当前测试主要用于 bring-up**
   当前主要完成基础功能验证，后续仍可继续补充更多输入模式与 corner case 测试。

## Files Modified / Added

本原型通常涉及以下新增或修改文件：

### RTL / Config

* `Mxfp.scala`
* `MxfpBall.scala`
* `configs/MxfpBallParam.scala`
* `configs/default.json`

### ISA / Test

* `55_mxfp.c`
* `mxfp_test.c`

### Integration points

* Ball 注册
* decoder 映射
* ISA include
* ball 配置映射

## Future Work

后续可以继续推进以下方向：

1. 扩展更多输入数据模式
2. 增加 corner case 测试
3. 完善 NaN / Inf / subnormal 的处理策略
4. 支持更多 MX 格式
5. 优化打包方式与参数化能力
6. 补充更系统的性能测试结果

## Summary

本 MXFP Ball 原型已经完成了从：

* 自定义指令
* Ball domain RTL
* Scratchpad 读写
* FP32 block 收集
* MXFP packed block 生成
* C 侧黄金模型比对

这一整条路径的验证。

当前版本已经能够作为后续 MXFP 扩展与优化的基础实现。

```
```
