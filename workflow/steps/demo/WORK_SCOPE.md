# Agent 工作范围和修改记录规范

## 工作路径限制

### Spec Agent

**允许的路径：**
- ✅ `arch/src/main/scala/prototype/<package>/<ball>/spec.md`（新建）

**禁止的路径：**
- ❌ 其他任何文件

### Code Agent

**允许的路径（新建）：**
1. **Chisel 模块**
   - `arch/src/main/scala/prototype/<package>/<ball>/XXXUnit.scala`
   - `arch/src/main/scala/prototype/<package>/<ball>/XXXCtrlUnit.scala`
   - `arch/src/main/scala/prototype/<package>/<ball>/XXXLoadUnit.scala`
   - `arch/src/main/scala/prototype/<package>/<ball>/XXXExUnit.scala`
   - `arch/src/main/scala/prototype/<package>/<ball>/XXXStoreUnit.scala`

2. **ISA API**
   - `bb-tests/workloads/lib/bbhw/isa/NN_xxx.c`（新建）

3. **测试用例**
   - `bb-tests/workloads/src/CTest/ctest_xxx_test.c`（新建）

**允许的路径（只能追加）：**
1. **ISA 注册**
   - `bb-tests/workloads/lib/bbhw/isa/isa.h`（只追加 enum）
   - `bb-tests/workloads/lib/bbhw/isa/isa.c`（只追加注册代码）
   - `bb-tests/workloads/lib/bbhw/isa/CMakeLists.txt`（只追加文件列表）

2. **系统注册**
   - `arch/src/main/scala/.../balldomain/DISA.scala`（只追加 BitPat）
   - `arch/src/main/scala/.../DomainDecoder.scala`（只追加解码条目）
   - `arch/src/main/scala/.../busRegister.scala`（只追加 Ball ID）
   - `arch/src/main/scala/.../rsRegister.scala`（只追加集成代码）

**禁止的路径：**
- ❌ 其他 Ball 的实现文件
- ❌ 已有的测试文件
- ❌ 系统核心文件（除了上述允许追加的文件）
- ❌ 配置文件
- ❌ 文档文件（除了 spec.md）

### Review Agent

**允许的路径（只读）：**
- ✅ code_agent 列出的所有修改文件（读取检查）

**禁止的路径：**
- ❌ 不在 code_agent 修改列表中的文件（避免误判）
- ❌ 写入任何文件（除非发现明显的拼写错误等小问题）

### Verify Agent

**允许的路径：**
- ✅ 读取所有必要的文件
- ✅ 生成测试报告文件
- ✅ 调用 workflow API（编译、测试）

**禁止的路径：**
- ❌ 修改源代码文件
- ❌ 修改测试用例

---

## 修改记录规范

### 为什么需要修改记录？

1. **防止误判**：review_agent 只检查本轮修改的文件，不会误判项目中已有的文件
2. **清晰追踪**：明确知道每个 agent 做了什么
3. **便于回滚**：出问题时可以快速定位
4. **责任明确**：每个文件的修改者清晰可见

### Spec Agent 的输出格式

```markdown
✅ 任务完成

本次修改的文件：
- arch/src/main/scala/prototype/nagisa/gelu/spec.md (新建)

文件说明：
- spec.md: GELU Ball 的设计规范，定义了输入输出、接口、指令语义等
```

### Code Agent 的输出格式

```markdown
✅ 任务完成

本次修改的文件：

【新建文件】
- arch/src/main/scala/prototype/nagisa/gelu/GELUUnit.scala
- arch/src/main/scala/prototype/nagisa/gelu/GELUCtrlUnit.scala
- arch/src/main/scala/prototype/nagisa/gelu/GELULoadUnit.scala
- arch/src/main/scala/prototype/nagisa/gelu/GELUExUnit.scala
- arch/src/main/scala/prototype/nagisa/gelu/GELUStoreUnit.scala
- bb-tests/workloads/lib/bbhw/isa/35_gelu.c
- bb-tests/workloads/src/CTest/ctest_gelu_test.c

【修改文件】（只追加，未删除/修改已有内容）
- bb-tests/workloads/lib/bbhw/isa/isa.h
  追加内容：enum GELU_BALL = 35

- bb-tests/workloads/lib/bbhw/isa/isa.c
  追加内容：注册 GELU_BALL 的指令函数

- bb-tests/workloads/lib/bbhw/isa/CMakeLists.txt
  追加内容：35_gelu.c

- arch/src/main/scala/examples/toy/balldomain/DISA.scala
  追加内容：BitPat("b100011") -> "gelu"

- arch/src/main/scala/examples/toy/balldomain/DomainDecoder.scala
  追加内容：case "gelu" => Module(new GELUUnit())

- arch/src/main/scala/examples/toy/busRegister.scala
  追加内容：GELUBall -> 35

- arch/src/main/scala/examples/toy/rsRegister.scala
  追加内容：集成 GELU Ball 到 RocketSystem

【未修改】
- 所有其他 Ball 的文件（relu, matmul 等）
- 系统核心文件
- 已有的测试文件

【文件说明】
- GELUUnit.scala: GELU Ball 顶层模块
- GELUCtrlUnit.scala: 控制单元，处理指令解析和状态机
- GELULoadUnit.scala: 加载单元，从 scratchpad 读取数据
- GELUExUnit.scala: 执行单元，计算 GELU 激活函数
- GELUStoreUnit.scala: 存储单元，将结果写回
- 35_gelu.c: GELU 指令的 C API 函数
- ctest_gelu_test.c: GELU 功能测试用例
```

### Review Agent 的检查流程

```markdown
📋 审查开始

## 第1步：获取修改文件列表

从 code_agent 的输出中提取：
【新建文件】
- arch/src/main/scala/prototype/nagisa/gelu/GELUUnit.scala
- ...

【修改文件】
- bb-tests/workloads/lib/bbhw/isa/isa.h (追加 enum)
- ...

## 第2步：检查新建文件（逐个检查）

✅ arch/src/main/scala/prototype/nagisa/gelu/GELUUnit.scala
- 继承了 Blink 接口
- 状态机逻辑完整
- 无 TODO 标记

✅ arch/src/main/scala/prototype/nagisa/gelu/GELUCtrlUnit.scala
- 控制逻辑完整
- 指令解析正确

...

## 第3步：检查修改文件（确认只追加）

✅ bb-tests/workloads/lib/bbhw/isa/isa.h
- 只在 enum 末尾追加了 GELU_BALL = 35
- 未修改已有的 enum 值

✅ bb-tests/workloads/lib/bbhw/isa/isa.c
- 只在注册数组末尾追加了新条目
- 未修改已有的注册代码

...

## 第4步：审查结论

✅ 审查通过

所有检查项已满足：
- RTL 实现完整（所有 Chisel 模块已实现）
- ISA API 完整（35_gelu.c 已实现）
- 系统注册完整（所有注册文件已正确追加）
- 测试用例完整（ctest_gelu_test.c 已编写）
- 未删除/修改已有代码（所有修改都是追加）

可以进入验证环节。
```

### 错误示例：没有提供文件列表

**Code Agent 的错误输出：**
```
完成了 GELU Ball 的实现。
```

**Review Agent 的响应：**
```
❌ 审查不通过 - 缺少修改记录

Code agent 没有提供本轮修改的文件列表。

无法确认：
1. 哪些文件是本轮新建的
2. 哪些文件是本轮修改的
3. 修改了哪些内容

请 code_agent 明确列出：
【新建文件】
- 文件1
- 文件2

【修改文件】
- 文件3 (追加了什么内容)
- 文件4 (追加了什么内容)

【未修改】
- 其他所有文件
```

---

## 检查清单

### Code Agent 自检清单

在完成任务前，确认：
- [ ] 我只修改了允许的路径下的文件
- [ ] 对于"只能追加"的文件，我只在末尾添加了新内容
- [ ] 我没有修改其他 Ball 的文件
- [ ] 我没有删除任何已有代码
- [ ] 我准备了完整的修改文件列表（新建/修改/未修改）
- [ ] 我为每个修改的文件提供了说明

### Review Agent 检查清单

在开始审查前，确认：
- [ ] Code agent 提供了修改文件列表
- [ ] 我只检查列表中的文件
- [ ] 我不检查不在列表中的文件（避免误判已有文件）

在审查时，确认：
- [ ] 所有新建文件是否完整
- [ ] 所有修改文件是否只追加（未删除/修改已有内容）
- [ ] 是否有文件不在允许的路径下
- [ ] 是否有删除/修改其他 Ball 的代码

---

## 工作范围违规处理

### 如果 Code Agent 修改了不允许的文件

```
❌ 审查不通过 - 工作范围违规

发现以下违规：
1. 修改了其他 Ball 的文件：arch/src/main/scala/prototype/nagisa/relu/ReluUnit.scala
   - Code agent 只能修改当前 Ball 的文件

修复建议：
- 撤销对其他 Ball 文件的修改
- 只修改当前 Ball 的文件
```

### 如果 Code Agent 删除了已有代码

```
❌ 审查不通过 - 删除了已有代码

发现以下问题：
1. bb-tests/workloads/lib/bbhw/isa/isa.h: 删除了已有的 enum RELU_BALL
   - 原有代码是正确的，不应该被删除

修复建议：
- 恢复被删除的代码
- 只在末尾追加新的 enum
```

### 如果 Review Agent 检查了不该检查的文件

```
⚠️ 审查流程错误

你检查了文件：arch/src/main/scala/prototype/nagisa/relu/ReluUnit.scala
但这个文件不在 code_agent 的修改列表中。

这可能是项目中已有的文件，不应该被检查。

请只检查 code_agent 明确列出的本轮修改的文件！
```

---

## 总结

**核心原则：**
1. **路径限制**：每个 agent 只能在指定路径下工作
2. **修改记录**：每个 agent 必须明确列出修改的文件
3. **只检查本轮**：review_agent 只检查本轮修改的文件
4. **只追加不修改**：对于共享文件（如 isa.h），只能追加不能修改

**好处：**
- ✅ 防止误判已有文件
- ✅ 清晰的责任划分
- ✅ 便于追踪和回滚
- ✅ 避免破坏已有代码
