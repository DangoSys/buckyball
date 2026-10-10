SECTIONS {
  . = 0;
  .text : { *(.text*) }
  ASSERT(SIZEOF(.text) <= 4096, "RVV instruction memory overflow")
  . = 0x40000000;
  .rodata : { *(.rodata*) *(.srodata*) }
  ASSERT(SIZEOF(.rodata) <= 4096, "RVV constant memory overflow")
  .data : { *(.data*) *(.sdata*) *(.tdata*) }
  .bss (NOLOAD) : { *(.bss*) *(.sbss*) *(.tbss*) *(COMMON) }
  ASSERT(SIZEOF(.data) == 0, "RVV kernels cannot define writable global data")
  ASSERT(SIZEOF(.bss) == 0, "RVV kernels cannot define writable global data")
  /DISCARD/ : { *(.eh_frame*) *(.comment*) *(.riscv.attributes*) }
}
