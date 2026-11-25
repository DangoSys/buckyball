/* An RVV operator image: a 24-byte header, the code, then the constants.
 *
 * The code runs at address 0 of the instruction memory and the constants at
 * 0x40000000 of the constant memory; the load addresses lay header, code and
 * constants out back to back, so `objcopy -O binary` writes the image that
 * mvin_operator loads. Link with `--defsym=operator_entry=<launch symbol>`. */
ENTRY(operator_entry)
SECTIONS {
  .text 0 : AT(24) { *(.text*) }
  .rodata 0x40000000 : AT(24 + SIZEOF(.text)) { *(.rodata*) *(.srodata*) . = ALIGN(4); }
  /* magic "RKV1", code bytes, entry, constant base, constant bytes, reserved */
  .header 0x80000000 : AT(0) {
    LONG(0x31564B52) LONG(SIZEOF(.text)) LONG(operator_entry)
    LONG(0x40000000) LONG(SIZEOF(.rodata)) LONG(0)
  }
  .data : { *(.data*) *(.sdata*) *(.tdata*) }
  .bss (NOLOAD) : { *(.bss*) *(.sbss*) *(.tbss*) *(COMMON) }
  /DISCARD/ : { *(.eh_frame*) *(.comment*) *(.riscv.attributes*) }
  ASSERT(SIZEOF(.text) <= 4096 && SIZEOF(.text) % 4 == 0, "RVV instruction memory overflow")
  ASSERT(operator_entry < SIZEOF(.text) && operator_entry % 4 == 0, "RVV operator entry outside its code")
  ASSERT(SIZEOF(.rodata) <= 4096, "RVV constant memory overflow")
  ASSERT(SIZEOF(.data) == 0 && SIZEOF(.bss) == 0, "RVV operators cannot define writable global data")
}
