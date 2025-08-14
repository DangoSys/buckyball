ENTRY(_start)

SECTIONS
{
  /* 代码段 */
  . = 0x80000000;
  .text : {
    *(.text.start)
    *(.text)
  }
  
  /* 只读数据段 */
  .rodata : {
    *(.rodata)
  }
  
  /* 数据段 */
  .data : {
    *(.data)
  }
  
  /* BSS段 */
  .bss : {
    *(.bss)
  }
  
  /* 定义_end符号供库使用 */
  _end = .;
  
  /* 栈顶定义 */
  . = 0x80010000;
  stack_top = .;
  __heap_end = .;
  
  /* 启动向量 - 考虑4字节偏移 */
  . = 0x1fffffc;
  .bootvec : {
    LONG(0)          /* 占位 */
    LONG(0x80000000) /* 实际跳转地址 */
  }
}