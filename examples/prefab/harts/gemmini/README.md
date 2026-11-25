# Gemmini hart

## 功能参考（Spike + libgemmini）

`ref/default.nix` 用 Nix 构建 Gemmini 的功能参考：riscv-isa-sim、riscv-pk 与 libgemmini 扩展，版本与 `thirdparty/gemmini` 所在的 Chipyard 版本一致，不依赖 Chipyard 和宿主机工具。在本目录下构建：

```bash
nix-build ref -o output/ref
```

运行裸机 ELF（HTIF `tohost` 退出）或经 pk 运行：

```bash
output/ref/bin/gemmini-spike /absolute/path/to/xxx-baremetal
output/ref/bin/gemmini-spike pk /absolute/path/to/xxx-pk
```

`gemmini-spike` 等价于 `spike --extlib=<ref>/lib/libgemmini.so --extension=gemmini`。模型的 RAX 包需要由 RISC-V Linux 中的 buddy-cli 加载，不能直接作为裸机 ELF 运行。
