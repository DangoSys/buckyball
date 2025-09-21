# FuncSim 工作流

BuckyBall 框架中的功能仿真工作流，提供快速的功能验证环境。

## API 使用说明

### `build`
**端点**: `POST /funcsim/build`

**功能**: 构建功能仿真器

**参数**: 无特定参数

**示例**:
```bash
bbdev funcsim --build
```

### `sim`
**端点**: `POST /funcsim/sim`

**功能**: 运行功能仿真

**参数**:
- **`binary`** - 要仿真的二进制文件路径
- **`ext`** - 扩展参数

**示例**:
```bash
# 基本仿真
bbdev funcsim --sim "--binary /path/to/test.elf"

# 带扩展参数
bbdev funcsim --sim "--binary /path/to/test.elf --ext additional_args"
```

## 典型工作流程

```bash
# 1. 构建功能仿真器
bbdev funcsim --build

# 2. 运行仿真
bbdev funcsim --sim "--binary ${buckyball}/bb-tests/workloads/build/src/CTest/ctest_basic-baremetal"
```

## 注意事项

- 只提供功能级别的仿真，不包含时序信息
- 确保二进制文件路径正确且可访问
