OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY {
  ram (rwx) : ORIGIN = 0x80000000, LENGTH = 128K
}
SECTIONS {
  . = ORIGIN(ram);
  .text : { KEEP(*(.text.init)) *(.text .text.*) } > ram
  .rodata : { *(.rodata .rodata.*) } > ram
  .data : ALIGN(16) { *(.data .data.*) *(.sdata .sdata.*) } > ram
  .bss (NOLOAD) : ALIGN(16) {
    __bss_start = .;
    *(.bss .bss.*) *(.sbss .sbss.*) *(COMMON)
    . = ALIGN(16);
    __bss_end = .;
  } > ram
  __stack_top = ORIGIN(ram) + LENGTH(ram);
  ASSERT(__bss_end <= __stack_top - 4096, "Firmware leaves less than 4 KiB stack")
}
