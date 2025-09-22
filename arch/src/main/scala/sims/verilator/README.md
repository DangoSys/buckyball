# Verilator 仿真配置

## 一、Overview

该目录包含了 BuckyBall 系统在 Verilator 平台上的仿真配置。Verilator 是一个开源的 Verilog/SystemVerilog 仿真器，能够将 RTL 代码编译为高性能的 C++ 仿真模型，提供快速的功能仿真和验证环境。

## 二、文件结构

```
verilator/
└── Elaborate.scala  - Verilator 详细化配置
```

## 三、核心实现

### Elaborate.scala

该文件实现了 BuckyBall 系统的 Verilog 生成和详细化过程：

```scala
object Elaborate extends App {
  val config = new examples.toy.BuckyBallToyConfig
  val params = config.toInstance

  ChiselStage.emitSystemVerilogFile(
    new chipyard.harness.TestHarness()(config.toInstance),
    firtoolOpts = args,
    args = Array.empty  // 直接传递命令行
  )
}
```
