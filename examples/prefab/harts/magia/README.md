# MAGIA hart

MAGIA 是 4×4 的 tile mesh，每个 tile 有一个 CV32E40P 主核、RedMulE（FP16 GEMM）、iDMA、
1 MiB L1，以及一个 Spatz 向量处理器（目前的软件栈没有用到它）。本 hart 用一个 RV64GC 的
主机核 Loom 运行程序，通过 custom-0 启动 mesh。Loom 的 FPU 带半精度（Zfh）：mesh 的操作数
和结果是 FP16，由 Loom 转换。

## 硬件接口（Loom 的 custom-0）

| funct7 | 含义 | 操作数 |
|---|---|---|
| 0 | 启动：复位 mesh，所有 tile 从入口开始执行 | rs1 = 物理入口地址 |
| 1 | 读状态：bit 0 完成，bit 63:32 为 tile 的汇总退出码 | rd |
| 2 | 取 mesh 的打印输出 | rd |
| 3 | 设置 mesh 的 DDR 窗口 | rs1 = 4 KiB 对齐的 DDR 地址，rs2 = 窗口号 |

mesh 是 32 位地址，只能看到 L2（`0xC000_0000` 起）。固件及其数据在 `0xCC00_0000` 起的
16 MiB 内按原地址访问；其上有三个窗口，由 Loom 指到 DDR 的任意位置：

| 窗口 | mesh 地址 | 大小 | 用途 |
|---|---|---|---|
| 0 | `0xD000_0000` | 512 MiB | W |
| 1 | `0xF000_0000` | 128 MiB | X |
| 2 | `0xF800_0000` | 128 MiB | Z |

复位后三个窗口映射到自身地址。

## 软件栈（`stack/`）

- `firmware/service.c`、`firmware/gemm.h`：tile 固件。一次启动做一个 GEMM 任务
  `Z = Z + X·W`（FP16）；任务描述在 `0xCC00_F000`（x、w、z、m、n、k、pitch）。16 个 tile
  按列分摊 W 和 Z：各自用 iDMA 把 X 和自己的列搬进 L1，跑 RedMulE，再把 Z 的列搬回。
  `pitch` 是 W、Z 每行在内存中的列数，任务可以只做更宽矩阵的一段列。每个 tile 至少分到
  两列（只有一列时 RedMulE 不计算），列数少时只用一部分 tile；一个任务至少两列。
- `runtime/mesh/gemm.c`：Loom 上的 `magia_gemm(x, w, z, m, n, k)`。设置窗口、写任务、启动并
  等待；一个 tile 放不下时按行、按列拆成多个任务。每次启动前清固件的启动标志
  （`0xCC01_0000`）。
- `runtime/reference/gemm.c`：软件里的 mesh 乘法（逐项 FP16 融合乘加），模型 recipe 在本机
  检查时使用，需要 AVX512-FP16。

`bbdev stack --build "--config magia/devices/magia/configs/<Config>.toml --device magia"`
产出 `mesh.bin`、`mesh.o`、`gemm.o`。

## 仿真器模型（`emu/`）

BEMU 里 `Mesh` 挂在 Loom 的 custom-0 上，按固件的语义直接完成任务（不逐条执行 tile 的
指令），所以只适用于运行 GEMM 服务固件的程序。一次任务的固定开销按 RTL 上小任务的实测
（约 3.7 万周期）取值；DMA 和 RedMulE 的速率还没有标定。

## 编译器与模型

Buddy 里有 `magia` dialect：`convert-linalg-to-magia` 把 FP32 的矩阵乘变成
`magia.gemm`（操作数转成 FP16；已经是 FP16 `[N, K]` 的权重原地读取），`lower-magia`
变成对 `magia_gemm` 的调用。模型 recipe 在 `stack/models/archs/magia/`：

```bash
cmake -S stack/models -B stack/models/build/magia -G Ninja -DARCH=magia \
  -DDESIGN_CMAKE=<design output>/stack/generated/design.cmake
ninja -C stack/models/build/magia <model>-model-check   # 本机，mesh 用软件参考
ninja -C stack/models/build/magia <model>-model-build   # 设备上的裸机程序
```

## 已知情况

- Loom 是框架里的 Rocket 核，约 13 周期一条指令，整模型的时间几乎都花在主机上。
- tile 访问 L2 的一个字约需一百个周期，固件和测试的数据要小。
- Verilator 约 0.3 kHz，一个测试要控制在 40 万周期以内。
