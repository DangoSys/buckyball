# Buckyball ISA Instruction Registration Mechanism

## Overview

This ISA library now supports a dynamic instruction registration mechanism, allowing you to easily add new custom instructions without modifying core code.

## How to Add New Instructions

### 1. Define New Instruction Type in Header File

Add new instructions to the `InstructionType` enum in `isa.h`:

```c
typedef enum {
  // Existing instructions...
  CUSTOM_ADD_FUNC7 = 50,    // New custom addition instruction
  CUSTOM_SUB_FUNC7 = 51,    // New custom subtraction instruction
} InstructionType;
```

### 2. Implement Instruction Execution Function

Create a new .c file or add to existing file:

```c
#ifndef __x86_64__
static void execute_custom_add(uint32_t rs1_val, uint32_t rs2_val) {
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 50, x0, %0, %1"
               : : "r"(rs1_val), "r"(rs2_val) : "memory");
}
#else
static void execute_custom_add_nop(uint32_t rs1_val, uint32_t rs2_val) {
  // No-op implementation on x86 platform
}
#endif
```

### 3. Register Instruction

Register new instruction in initialization code:

```c
void init_custom_instructions(void) {
#ifndef __x86_64__
  register_instruction(CUSTOM_ADD_FUNC7, execute_custom_add);
#else
  register_instruction(CUSTOM_ADD_FUNC7, execute_custom_add_nop);
#endif
}
```

### 4. Use Instruction

```c
BuckyballInstruction inst = build_instruction(CUSTOM_ADD_FUNC7);
InstructionBuilder builder = create_builder(&inst, CUSTOM_ADD_FUNC7);

// Set parameters
builder.set.rs1(builder.set.builder_ptr, "field_name", value1);
builder.set.rs2(builder.set.builder_ptr, "field_name", value2);

// Execute instruction
execute_builder(builder);
```

## Advantages

1. **Modularity**: New instructions can be implemented in separate files
2. **Extensibility**: Add new instructions without modifying core code
3. **Platform Compatibility**: Automatically handles differences between x86 and RISC-V platforms
4. **Type Safety**: Compile-time checking of instruction types
5. **Performance**: Minimal runtime lookup overhead

## Notes

- func7 values must be in the range 0-127
- Each instruction type can only be registered once
- Supports up to 16 instructions (can be adjusted by modifying MAX_INSTRUCTIONS)
- On RISC-V platforms, func7 values in inline assembly must be compile-time constants
