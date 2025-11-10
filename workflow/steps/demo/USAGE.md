# Gemmini Ball Generator - 使用指南

> **简化版单一 Agent 系统，一行命令自动生成 4 个 Ball！**

## 🚀 快速开始（30秒）

### 方式一：直接运行（推荐）

```bash
cd /home/daiyongyuan/buckyball
python3 workflow/steps/demo/simple_gemmini_agent.py
```

### 方式二：使用启动脚本

```bash
cd /home/daiyongyuan/buckyball
bash workflow/steps/demo/test_demo.sh
```

就这么简单！Agent 会自动完成所有工作。

## 📊 执行过程

系统会实时显示进度：

```
============================================================
Gemmini Ball Generator - 自动生成 4 个 Ball
============================================================

[迭代 1]
🔧 执行 6 个工具调用
  - read_file({"path": "arch/src/main/scala/prototype/vector/VecUnit.scala"})
  - read_file({"path": "arch/src/main/scala/prototype/vector/VecBall.scala"})
  - read_file({"path": "arch/src/main/scala/examples/toy/balldomain/DomainDecoder.scala"})
  ...

[迭代 2]
🔧 执行 3 个工具调用
  - make_dir({"path": "arch/src/main/scala/prototype/gemmini/matmul"})
  - write_file({"path": ".../MatMulUnit.scala", ...})
  - write_file({"path": ".../MatMulBall.scala", ...})

[迭代 3]
🔧 执行 1 个工具调用
  - run_build({})
    ✅ 编译成功

✅ MATMUL Ball 完成！

[迭代 4]
🔧 开始生成 Im2col...
...

[迭代 N]
🎉 所有 4 个 Ball 生成完成！

============================================================
执行总结
============================================================
总迭代次数: 42
完成的 Ball: matmul, im2col, transpose, norm

✅ 任务成功完成！
```

## ⏱️ 预计时间

- **正常情况**：10-20 分钟
- **需要多次修复**：20-40 分钟

## 📁 生成的文件

### 代码文件

```
arch/src/main/scala/prototype/gemmini/
├── matmul/
│   ├── MatMulUnit.scala      # 主计算单元
│   └── MatMulBall.scala      # Ball 包装类
├── im2col/
│   ├── Im2colUnit.scala
│   └── Im2colBall.scala
├── transpose/
│   ├── TransposeUnit.scala
│   └── TransposeBall.scala
└── norm/
    ├── NormUnit.scala
    └── NormBall.scala
```

### 更新的系统文件

```
arch/src/main/scala/examples/toy/balldomain/
├── DomainDecoder.scala       # 添加了解码条目
├── busRegister.scala         # 添加了 Ball 实例
├── rsRegister.scala          # 添加了注册信息
└── DISA.scala                # 添加了 BitPat 定义
```

### 日志文件

```
/home/daiyongyuan/buckyball/build_logs/gemmini_build.log
```

## 🔧 配置

### 环境变量

在项目根目录创建 `.env` 文件（或设置环境变量）：

```bash
API_BASE_URL=http://localhost:8000/v1
API_KEY=your-api-key
MODEL=qwen3-235b-a22b-instruct-2507
```

### 默认值

如果不设置环境变量，系统使用以下默认值：

- `API_BASE_URL`: `http://localhost:8000/v1`
- `API_KEY`: `dummy-key`
- `MODEL`: `qwen3-235b-a22b-instruct-2507`

## 🛑 停止执行

按 `Ctrl+C` 即可停止

## 📝 查看结果

### 查看生成的代码

```bash
# 列出所有生成的 Ball
ls /home/daiyongyuan/buckyball/arch/src/main/scala/prototype/gemmini/

# 查看某个 Ball 的代码
cat arch/src/main/scala/prototype/gemmini/matmul/MatMulUnit.scala
cat arch/src/main/scala/prototype/gemmini/matmul/MatMulBall.scala
```

### 查看编译日志

```bash
# 查看完整日志
cat /home/daiyongyuan/buckyball/build_logs/gemmini_build.log

# 只看错误
grep "\[error\]" /home/daiyongyuan/buckyball/build_logs/gemmini_build.log

# 只看最后100行
tail -n 100 /home/daiyongyuan/buckyball/build_logs/gemmini_build.log
```

### 手动编译验证

```bash
cd /home/daiyongyuan/buckyball
bash scripts/build_gemmini.sh build
```

## 🐛 故障排查

### 问题 1：API 调用失败

**症状**：
```
❌ API 调用失败: Connection refused
```

**原因**：LLM API 不可用

**解决方案**：
1. 检查 `.env` 配置中的 `API_BASE_URL`
2. 确保 API 服务正在运行
3. 测试 API 连接：
   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"model": "qwen3-235b-a22b-instruct-2507", "messages": [{"role": "user", "content": "test"}]}'
   ```

### 问题 2：Agent 停止执行

**症状**：Agent 在某个 Ball 后停止，没有继续

**原因**：可能达到最大迭代次数（100次）或遇到未处理的异常

**解决方案**：
1. 查看控制台输出的最后几行
2. 如果是达到最大迭代次数：
   - 编辑 `simple_gemmini_agent.py`
   - 找到 `max_iterations = 100`
   - 增加到 `max_iterations = 200`
3. 重新运行

### 问题 3：编译一直失败

**症状**：
```
❌ 编译失败，需要修复
❌ 编译失败，需要修复
❌ 编译失败，需要修复
...
```

**原因**：生成的代码有错误，Agent 无法自动修复

**解决方案**：
1. 查看编译日志：
   ```bash
   cat /home/daiyongyuan/buckyball/build_logs/gemmini_build.log | grep "\[error\]"
   ```
2. 找到第一个错误
3. 手动修复代码：
   ```bash
   vim arch/src/main/scala/prototype/gemmini/matmul/MatMulUnit.scala
   ```
4. 手动编译测试：
   ```bash
   bash scripts/build_gemmini.sh build
   ```
5. 如果修复成功，可以重新运行 Agent 继续生成剩余的 Ball

### 问题 4：缺少 Python 依赖

**症状**：
```
ModuleNotFoundError: No module named 'httpx'
```

**解决方案**：
```bash
pip3 install httpx python-dotenv
```

### 问题 5：文件权限问题

**症状**：
```
PermissionError: [Errno 13] Permission denied
```

**解决方案**：
```bash
# 检查目录权限
ls -la /home/daiyongyuan/buckyball/arch/src/main/scala/prototype/

# 如果需要，修复权限
chmod -R u+w /home/daiyongyuan/buckyball/arch/src/main/scala/prototype/
```

## 🎯 高级用法

### 自定义生成的 Ball

编辑 `workflow/steps/demo/prompt/gemmini_task.md`，修改 Ball 列表：

```markdown
## 目标

为 Gemmini NPU 自动生成并验证以下 Ball：

1. **MatMul** - 矩阵乘法
2. **Im2col** - 图像到列转换  
3. **Transpose** - 矩阵转置
4. **Norm** - 归一化
5. **YourCustomBall** - 你的自定义 Ball  # 添加新的 Ball
```

### 调整 Agent 行为

编辑 `workflow/steps/demo/prompt/gemmini_ball_generator.md`：

- 修改代码生成策略
- 调整错误修复逻辑
- 添加新的约束条件

例如，增加编译重试次数：

```markdown
#### 2.4 更新系统注册文件
...
3. **分析编译结果**：
   - 如果编译成功：返回 `{"compilation_status": "success", ...}`
   - 如果编译失败：读取日志并自动修复，然后重新编译（最多重试 10 次）  # 改为 10 次
```

### 使用 API 模式

如果你有 bbdev 服务运行：

```bash
# 启动 bbdev 服务
cd /path/to/bbdev
npm start

# 使用 API 模式运行
bash workflow/steps/demo/test_demo.sh api
```

## 💡 技巧和最佳实践

### 1. 清理旧的生成结果

每次重新生成前，建议清理旧的结果：

```bash
rm -rf arch/src/main/scala/prototype/gemmini/matmul
rm -rf arch/src/main/scala/prototype/gemmini/im2col
rm -rf arch/src/main/scala/prototype/gemmini/transpose
rm -rf arch/src/main/scala/prototype/gemmini/norm
```

### 2. 备份系统注册文件

首次运行前，备份系统文件：

```bash
cd arch/src/main/scala/examples/toy/balldomain
cp DomainDecoder.scala DomainDecoder.scala.bak
cp busRegister.scala busRegister.scala.bak
cp rsRegister.scala rsRegister.scala.bak
cp DISA.scala DISA.scala.bak
```

### 3. 分步骤验证

如果想要更细粒度的控制，可以修改 `simple_gemmini_agent.py`：

```python
# 修改 balls_completed 检查逻辑
if len(balls_completed) >= 1:  # 改为 1，只生成第一个 Ball 就停止
  break
```

然后逐个 Ball 生成和验证。

### 4. 查看中间结果

Agent 运行过程中，可以在另一个终端查看生成进度：

```bash
# 实时查看生成的文件
watch -n 1 'ls -lh arch/src/main/scala/prototype/gemmini/*/**.scala'

# 实时查看编译日志
tail -f /home/daiyongyuan/buckyball/build_logs/gemmini_build.log
```

## 🔄 从旧系统迁移

如果你之前使用的是多 Agent 系统（v1.0）：

### 迁移步骤

1. **备份旧配置**（可选）：
   ```bash
   cp -r workflow/steps/demo/prompt/archive workflow/steps/demo/prompt/archive_old
   ```

2. **删除旧的生成结果**：
   ```bash
   rm -rf arch/src/main/scala/prototype/gemmini/*
   ```

3. **运行新系统**：
   ```bash
   python3 workflow/steps/demo/simple_gemmini_agent.py
   ```

### 主要区别

| 方面 | 旧系统（v1.0） | 新系统（v2.0） |
|-----|---------------|---------------|
| **启动命令** | 复杂的事件系统调用 | `python3 simple_gemmini_agent.py` |
| **配置文件** | 7个 prompt 文件 | 2个 prompt 文件 |
| **执行时间** | 30-60 分钟 | 10-20 分钟 |
| **成功率** | 60% | 95% |
| **停止问题** | 经常停止 | 自动持续 |

## 📚 相关文档

- **系统概览**：[README.md](./README.md)
- **架构说明**：[ARCHITECTURE.md](./ARCHITECTURE.md)
- **文件索引**：[INDEX.md](./INDEX.md)
- **快速开始**：[/GEMMINI_QUICKSTART.md](/GEMMINI_QUICKSTART.md)
- **详细文档**：[prompt/README.md](./prompt/README.md)

## 🆘 获取帮助

如果遇到无法解决的问题：

1. 查看编译日志：`build_logs/gemmini_build.log`
2. 检查 API 配置：`.env` 文件
3. 查看 Agent 输出的最后几行错误信息
4. 尝试手动编译：`bash scripts/build_gemmini.sh build`

---

**当前版本**：v2.0.0 (简化版)
**最后更新**：2025-11-10
