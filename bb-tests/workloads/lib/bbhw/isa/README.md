# Buckyball ISA 指令注册机制

## 概述

这个ISA库现在支持动态指令注册机制，允许你轻松添加新的自定义指令而无需修改核心代码。

## 如何添加新指令

### 1. 在头文件中定义新的指令类型

在 `isa.h` 中的 `InstructionType` 枚举中添加新的指令：

```c
typedef enum {
  // 现有指令...
  CUSTOM_ADD_FUNC7 = 50,    // 新的自定义加法指令
  CUSTOM_SUB_FUNC7 = 51,    // 新的自定义减法指令
} InstructionType;
```

### 2. 实现指令执行函数

创建一个新的.c文件或在现有文件中添加：

```c
#ifndef __x86_64__
static void execute_custom_add(uint32_t rs1_val, uint32_t rs2_val) {
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 50, x0, %0, %1"
               : : "r"(rs1_val), "r"(rs2_val) : "memory");
}
#else
static void execute_custom_add_nop(uint32_t rs1_val, uint32_t rs2_val) {
  // x86平台下的空实现
}
#endif
```

### 3. 注册指令

在初始化代码中注册新指令：

```c
void init_custom_instructions(void) {
#ifndef __x86_64__
  register_instruction(CUSTOM_ADD_FUNC7, execute_custom_add);
#else
  register_instruction(CUSTOM_ADD_FUNC7, execute_custom_add_nop);
#endif
}
```

### 4. 使用指令

```c
BuckyballInstruction inst = build_instruction(CUSTOM_ADD_FUNC7);
InstructionBuilder builder = create_builder(&inst, CUSTOM_ADD_FUNC7);

// 设置参数
builder.set.rs1(builder.set.builder_ptr, "field_name", value1);
builder.set.rs2(builder.set.builder_ptr, "field_name", value2);

// 执行指令
execute_builder(builder);
```

## 优势

1. **模块化**：新指令可以在单独的文件中实现
2. **可扩展**：无需修改核心代码就能添加新指令
3. **平台兼容**：自动处理x86和RISC-V平台的差异
4. **类型安全**：编译时检查指令类型
5. **性能**：运行时查找开销很小

## 注意事项

- func7值必须在0-127范围内
- 每个指令类型只能注册一次
- 最多支持16个指令（可通过修改MAX_INSTRUCTIONS调整）
- 在RISC-V平台上，内联汇编中的func7值必须是编译时常量
