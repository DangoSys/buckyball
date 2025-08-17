## 项目目录说明

<!-- toc -->

### 根目录
- `arch/` - Scala编写的架构设计文档和代码
- `bb-test/` - 测试相关
- `compiler/` - **subtree** MLIR-based编译器，含LLVM等submodule
- `docs/` - 文档目录
- `scripts/` - 初始化脚本
- `sim/` - **submodule** RISC-V模拟器(spike)
- `thirdparty/` - **submodules** 第三方依赖
  - `chipyard/` - SoC设计框架
  - `circt/` - CIRCT电路编译器
- `tools/` - **submodules** 工具
  - `motia/` - 后端服务框架
  - `vistools/` - 可视化工具
- `workflow/` - CI/CD工作流配置

### compiler/ 内部结构
- `llvm/` - **submodule** LLVM主项目
- `thirdparty/` - **submodules** 编译器依赖
  - `mimalloc/` - 内存分配器
  - `riscv-gnu-toolchain/` - RISC-V工具链
- `examples/` - 各种MLIR示例和模型
- `frontend/` - 前端代码生成
- `midend/` - 中端优化
- `tools/` - 编译工具(buddy-opt等)

注: **submodule**需独立更新，`subtree`随主仓库同步
