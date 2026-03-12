# TraceBall — 调试追踪 Ball

## 概述

TraceBall 是一个不做计算的特殊 Ball，通过 Buckyball 指令通道提供运行时调试能力。它有两个核心功能：

1. **Cycle Counter（计数器管理）**— 通过指令 set/release 多个独立的 cycle 计数器，用于测量任意代码区间的执行周期
2. **Bank Backdoor（SRAM 后门读写）**— 通过 DPI-C 注入数据写入 SRAM bank，或读取 SRAM bank 数据通过 DPI-C 输出

所有 DPI-C 接口仅存在于 TraceBall 内部，不影响其他模块。

---

## 指令编码

TraceBall 使用 **两个 funct7 编码**。

### 指令 1：`bdb_counter` (funct7 = 48, 0x30)

Cycle counter 管理。**不访问 SRAM，不需要 bank 端口，1 cycle 完成。**

rs1 布局：
- rs1 = 任意（不使用，不需要设 BB_RD0/BB_WR 标志）

rs2 布局（64-bit）：
```
rs2[3:0]   = subcmd    子命令 (0=START, 1=STOP, 2=READ)
rs2[7:4]   = ctr_id    计数器编号 (0-15，最多 16 个独立计数器)
rs2[63:8]  = payload
```

子命令定义：

| subcmd | 名称 | 行为 | payload 含义 |
|--------|------|------|-------------|
| 0 | `CTR_START` | 启动计数器 ctr_id，记录当前 cycle 为起始点 | payload = tag（用户自定义标签，会输出到 trace） |
| 1 | `CTR_STOP` | 停止计数器 ctr_id，输出 elapsed cycles 到 DPI-C trace，然后释放计数器 | payload = 忽略 |
| 2 | `CTR_READ` | 读取计数器 ctr_id 当前值（不停止），输出到 DPI-C trace | payload = 忽略 |

DPI-C 输出格式（写入 bdb.log）：
```
[CTRACE] CTR_START  ctr=0 tag=0xDEAD cycle=10042
[CTRACE] CTR_STOP   ctr=0 tag=0xDEAD elapsed=387 cycle=10429
[CTRACE] CTR_READ   ctr=0 current=200 cycle=10242
```

### 指令 2：`bdb_backdoor` (funct7 = 49, 0x31)

SRAM 后门读写，**所有参数（bank_id, row, data）由 DPI-C 提供**。**需要 bank 端口（inBW=1, outBW=1）。**

rs1 布局：
```
rs1[45]     = BB_RD0    读模式：从 DPI-C 获取地址，读 SRAM，输出数据到 DPI-C
rs1[47]     = BB_WR     写模式：从 DPI-C 获取地址+数据，写 SRAM
rs1[63:48]  = iter      操作次数（0 = 单次，>0 = 循环 iter 次）
```

rs2 布局：
- rs2 = 任意（不使用）

操作模式：

| rs1 flag | 行为 | DPI-C 交互 |
|----------|------|-----------|
| BB_RD0 | 读模式 | RTL 调用 `dpi_backdoor_get_read_addr()` 获取 (bank_id, row)，读 SRAM，调用 `dpi_backdoor_put_read_data()` 输出数据 |
| BB_WR | 写模式 | RTL 调用 `dpi_backdoor_get_write_req()` 获取 (bank_id, row, data)，写 SRAM |

DPI-C 输出格式：
```
[BANK-TRACE] BACKDOOR_READ  bank=2 row=5 data=0x00010002000300040005000600070008
[BANK-TRACE] BACKDOOR_WRITE bank=3 row=10 data=0xDEADBEEFDEADBEEFDEADBEEFDEADBEEF
```


## 使用示例

### 测量算子执行周期

```c
// 测量一次 matmul 的 cycle
bdb_counter_start(0, 0xA001);         // 计数器 0，tag=matmul
bb_mul_warp16(A, B, C, 16);
bb_fence();
bdb_counter_stop(0);                  // 输出 elapsed

// 测量嵌套区间
bdb_counter_start(0, 0xB001);         // 外层：整个 conv
  bdb_counter_start(1, 0xB002);       // 内层：im2col
  bb_im2col(...);
  bb_fence();
  bdb_counter_stop(1);

  bdb_counter_start(2, 0xB003);       // 内层：matmul
  bb_mul_warp16(...);
  bb_fence();
  bdb_counter_stop(2);
bdb_counter_stop(0);                  // 外层结束
```

bdb.log 输出：
```
[CTRACE] CTR_START  ctr=0 tag=0xB001 cycle=0
[CTRACE] CTR_START  ctr=1 tag=0xB002 cycle=0
[CTRACE] CTR_STOP   ctr=1 tag=0xB002 elapsed=150 cycle=150
[CTRACE] CTR_START  ctr=2 tag=0xB003 cycle=0
[CTRACE] CTR_STOP   ctr=2 tag=0xB003 elapsed=300 cycle=300
[CTRACE] CTR_STOP   ctr=0 tag=0xB001 elapsed=456 cycle=456
```

### SRAM 后门注入测试数据

TraceBall内有一个私有bank，不会被配置看到

```c
// 不走 DMA，直接通过 DPI-C 注入测试数据到 bank 0
// （C++ 先通过 DPI-C 注入数据到TraceBall内的bank）
bb_alloc(0, 1, 1)
bdb_backdoor_mvin(16);     // 注入 16 行到 私有bank
bdb_backdoor_write(0, 16);     // 将私有bank的数据 注入 16 行到 bank 0

// 跑 Transpose
bb_transpose(0, 1, 16);
// 检查部分结果
bdb_backdoor_peek(0, 15);      // 检查最后一行

// 读出全部结果
bdb_backdoor_read(1, 16);     // dump bank 1 的 16 行到 trace
```



注意，所有RTL修改都在traceball内部，不允许动外部bank等地方
