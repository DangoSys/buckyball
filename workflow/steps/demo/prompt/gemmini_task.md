# Gemmini NPU Ball 自动生成任务

## 目标

为 Gemmini NPU 自动生成并验证 **4 个计算 Ball**：

1. **MatMul** - 矩阵乘法
2. **Im2col** - 图像到列转换
3. **Transpose** - 矩阵转置
4. **Norm** - 归一化

## 成功标准

✅ 所有 4 个 Ball 的代码生成完成
✅ 所有代码能够通过 `bash {/home/daiyongyuan/buckyball/scripts/build_gemmini.sh} build` 编译
✅ 无编译错误

## 工作流程

对于每个 Ball，按以下顺序执行：

### 1. 学习阶段
- 读取参考代码：`prototype/vector/VecUnit.scala` 和 `prototype/vector/VecBall.scala`
- 读取系统注册文件：`DomainDecoder.scala`、`busRegister.scala`、`rsRegister.scala`、`DISA.scala`
- 理解 Ball 的结构和接口规范

### 2. 生成阶段
为每个 Ball 生成以下文件：
- `arch/src/main/scala/prototype/generated/<ball>/<BallName>Unit.scala`
- `arch/src/main/scala/prototype/generated/<ball>/<BallName>Ball.scala`
- 更新系统注册文件（追加解码、注册信息）

### 3. 验证阶段
- 立即运行：`bash /home/daiyongyuan/buckyball/scripts/build_gemmini.sh build`
- 读取日志：`/home/daiyongyuan/buckyball/build_logs/gemmini_build.log`
- 如果编译失败：
  - 分析错误信息
  - 自动修复代码
  - 重新编译（最多重试 5 次）

### 4. 继续下一个
- 编译成功后，立即开始下一个 Ball
- 按顺序：matmul → im2col → transpose → norm

## 重要规则

⚠️ **必须按顺序完成所有 4 个 Ball**
⚠️ **编译失败必须自动修复，不能停止**
⚠️ **不能只生成部分文件**
⚠️ **每个 Ball 必须编译成功才能继续下一个**

## 参考代码位置

- 参考实现：`arch/src/main/scala/prototype/vector/`
- 系统注册：`arch/src/main/scala/examples/toy/balldomain/`
- 目标位置：`arch/src/main/scala/prototype/generated/`

## 立即开始

**第一步**：为 `matmul` Ball 生成代码并验证编译
