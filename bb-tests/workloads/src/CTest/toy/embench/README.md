# Embench on Buckyball

## 目录结构

```
bb-tests/workloads/src/CTest/toy/embench/
├── support/                  # 公共支持文件
│   ├── main.c               # 统一 main（调 initialise_board → benchmark → verify）
│   ├── beebsc.c/h           # rand/malloc 等简化 libc
│   ├── support.h            # 接口定义（始终 include boardsupport.h）
│   ├── boardsupport.c/h     # Buckyball 适配（空 init + start/stop_trigger）
│   └── chipsupport.c/h      # Buckyball 适配（空 init）
└── src/                      # 19 个 benchmark
    ├── aha-mont64/  crc32/  cubic/  edn/  huffbench/  matmult-int/
    ├── minver/  nbody/  nettle-aes/  nettle-sha256/  nsichneu/
    ├── picojpeg/  qrduino/  sglib-combined/  slre/  st/
    └── statemate/  ud/  wikisort/
```

## 编译调用链

```
toy/CMakeLists.txt
  └── add_embench_target(<bmark>)            # 总入口
        ├── add_embench_singlecore_target()   # 单核 → crt0.S
        └── add_embench_multicore_target()    # 多核 → start.S + -DMULTICORE=3
              └── ${ELF_CC} ${EMBENCH_C_FLAGS} -o ... \
                    <crt0.S | start.S> ${EMBENCH_SUPPORT_SOURCES} <bmark/*.c>

EMBENCH_C_FLAGS = ${C_FLAGS} -ffunction-sections -fdata-sections \
                  -Wl,--gc-sections -I${EMBENCH_SUPPORT_DIR}

C_FLAGS = -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany \
          -fno-builtin-printf -specs=nano.specs -specs=nosys.specs \
          -nostartfiles -Wl,-T,bbsim.ld -I.

EMBENCH_SUPPORT_SOURCES = main.c beebsc.c boardsupport.c chipsupport.c
```

## 最终编译命令

### 单核裸机

```bash
riscv64-unknown-elf-gcc \
  -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany \
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles \
  -Wl,-T,bb-tests/workloads/src/CTest/toy/bbsim.ld \
  -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -I bb-tests/workloads/src/CTest/toy \
  -I bb-tests/workloads/src/CTest/toy/embench/support \
  -o embench_<bmark>_singlecore-baremetal \
  bb-tests/workloads/src/CTest/toy/crt0.S \
  bb-tests/workloads/src/CTest/toy/embench/support/{main,beebsc,boardsupport,chipsupport}.c \
  bb-tests/workloads/src/CTest/toy/embench/src/<bmark>/*.c
```

### 多核裸机

差异：`crt0.S` → `start.S`，新增 `-DMULTICORE=3`，输出名 `*_multicore-baremetal`。

## 运行流程

```bash
# 1. 进入 nix 环境
cd /home/dean/buckyball
nix develop

# 2. 构建所有 embench (18/19 个，cubic 因 long double 问题排除)
cd bb-tests
rm -rf build && mkdir build && cd build
cmake ../workloads
make embench-all -j$(nproc)

# 或构建单个 benchmark
make embench_crc32                # crc32 单核+多核

# 3. Verilator 仿真（需通过 MCP 工具）
bbdev_verilator_run(
  binary="embench_crc32_singlecore-baremetal",
  config="sims.verilator.BuckyballToyVerilatorConfig"
)
```

## 构建结果

✅ **18/19 个 benchmark 编译成功** (36 个二进制文件)

| Benchmark | singlecore | multicore | 状态 |
|-----------|-----------|-----------|------|
| aha-mont64 | ✅ | ✅ | 成功 |
| crc32 | ✅ | ✅ | 成功 |
| cubic | ❌ | ❌ | long double 重定位溢出 |
| edn | ✅ | ✅ | 成功 |
| huffbench | ✅ | ✅ | 成功 |
| matmult-int | ✅ | ✅ | 成功 |
| minver | ✅ | ✅ | 成功 |
| nbody | ✅ | ✅ | 成功 |
| nettle-aes | ✅ | ✅ | 成功 |
| nettle-sha256 | ✅ | ✅ | 成功 |
| nsichneu | ✅ | ✅ | 成功 |
| picojpeg | ✅ | ✅ | 成功 |
| qrduino | ✅ | ✅ | 成功 |
| sglib-combined | ✅ | ✅ | 成功 |
| slre | ✅ | ✅ | 成功 |
| st | ✅ | ✅ | 成功 |
| statemate | ✅ | ✅ | 成功 |
| ud | ✅ | ✅ | 成功 |
| wikisort | ✅ | ✅ | 成功 |

输出位置：`bb-tests/build/src/CTest/toy/embench_*-baremetal`

输出格式：`ELF 64-bit LSB executable, UCB RISC-V, RVC, double-float ABI, statically linked`

## 关键改动 vs Embench 原版

| 项 | 原版 | Buckyball |
|----|------|-----------|
| Specs | `htif_nano.specs` | `nano.specs + nosys.specs` |
| ABI | `-mabi=lp64d` | 默认（rv64gc） |
| 启动 | HTIF | `crt0.S` / `start.S` |
| 链接脚本 | specs 内置 | `bbsim.ld`（0x80000000） |
| boardsupport | ri5cyverilator 配置 | 空 init + 编译屏障 trigger |
| 编译方式 | `-c` 分离编译 + 链接 | 一步直接生成 ELF |
| 多核 | 不支持 | `-DMULTICORE=3` + `start.S` |
