OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS
{
  . = 0x80000000;
  .text.start : { *(.text.start) }
  . = 0x80000200;
  .text.trap : { *(.text.trap) }
  . = 0x80000400;
  .text.tests : { *(.text.tests) }
  . = 0x800040c0;
  .data : { *(.data) }
  /DISCARD/ : { *(.riscv.attributes) *(.comment) }
}
