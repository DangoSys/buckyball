# FireSim 工作流

BuckyBall 框架中的 FireSim FPGA 仿真工作流，提供基于 FPGA 的硬件仿真环境。

## API 使用说明

### `buildbitstream`
**端点**: `POST /firesim/buildbitstream`

**功能**: 构建 FPGA 比特流文件

**参数**: 无特定参数

**示例**:
```bash
bbdev firesim --buildbitstream
```

### `infrasetup`
**端点**: `POST /firesim/infrasetup`

**功能**: 设置 FireSim 基础设施

**参数**: 无特定参数

**示例**:
```bash
bbdev firesim --infrasetup
```

### `runworkload`
**端点**: `POST /firesim/runworkload`

**功能**: 在 FireSim 上运行工作负载

**参数**: 无特定参数

**示例**:
```bash
bbdev firesim --runworkload
```

## 典型工作流程

```bash
# 1. 构建比特流
bbdev firesim --buildbitstream

# 2. 设置基础设施
bbdev firesim --infrasetup

# 3. 运行工作负载
bbdev firesim --runworkload
```

## 注意事项

- 比特流构建需要数小时时间
- infrasetup 需要配置云计算资源
