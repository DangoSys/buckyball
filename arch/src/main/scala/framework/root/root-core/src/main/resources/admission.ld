OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS
{
  . = 0x80000000;
  .text.start : { *(.text.start) }
  . = 0x80000200;
  .text.trap : { *(.text.trap) }
  . = 0x800040c0;
  .data : { *(.data) }
  . = 0x80006000;
  .pt.root : { *(.pt.root) }
  . = 0x80007000;
  .pt.mid : { *(.pt.mid) }
  . = 0x80008000;
  .pt.leaf : { *(.pt.leaf) }
  /DISCARD/ : { *(.riscv.attributes) *(.comment) }
}
