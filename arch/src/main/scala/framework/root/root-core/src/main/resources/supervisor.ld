OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS
{
  . = 0x80000000;
  .text.start : { *(.text.start) }
  . = 0x80002000;
  .text.trap : { *(.text.trap) }
  . = 0x80004000;
  .text.supervisor : { *(.text.supervisor) }
  . = 0x80006000;
  .text.user : { *(.text.user) }
  . = 0x80008000;
  .data : { *(.data) }
  . = 0x8000a000;
  .text.boundary : { *(.text.boundary) }
  . = 0x80010000;
  .pagetables : { *(.pagetables) }
  /DISCARD/ : { *(.riscv.attributes) *(.comment) }
}
