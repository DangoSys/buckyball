/* The operator image of the GP domain: a 24-byte header, the code, then the
 * constants. The code runs at address 0 of the instruction memory and the
 * constants at 0x40000000 of the constant memory; the load addresses lay
 * header, code and constants out back to back, so `objcopy -O binary` writes
 * the image that mvin_operator loads. The entry `operator` runs operator a7; the
 * operators reach memory through gp_space, the GP address space from 0.
 * Link with `--defsym=operator_text_bytes=<instruction memory bytes>`. */
ENTRY(operator)
gp_space = 0;
SECTIONS {
  .text 0 : AT(24) { *(.text*) . = ALIGN(8); }
  .rodata 0x40000000 : AT(24 + SIZEOF(.text)) { *(.rodata*) *(.srodata*) . = ALIGN(4); }
  /* magic "RKV1", code bytes, entry, constant base, constant bytes, reserved */
  .header 0x80000000 : AT(0) {
    LONG(0x31564B52) LONG(SIZEOF(.text)) LONG(operator)
    LONG(0x40000000) LONG(SIZEOF(.rodata)) LONG(0)
  }
  .data : { *(.data*) *(.sdata*) *(.tdata*) }
  .bss (NOLOAD) : { *(.bss*) *(.sbss*) *(.tbss*) *(COMMON) }
  /DISCARD/ : { *(.eh_frame*) *(.comment*) *(.riscv.attributes*) }
  ASSERT(SIZEOF(.text) <= operator_text_bytes, "GP instruction memory overflow")
  ASSERT(SIZEOF(.rodata) <= 4096, "GP constant memory overflow")
  ASSERT(SIZEOF(.data) == 0 && SIZEOF(.bss) == 0, "GP operators cannot define writable global data")
}
