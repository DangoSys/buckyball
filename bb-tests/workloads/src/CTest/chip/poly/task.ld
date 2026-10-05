OUTPUT_ARCH("riscv")
ENTRY(_start)
SECTIONS {
  . = 0x80000000;
  .text : { *(.text.init) *(.text.startup*) *(.text*) }
  .rodata : { *(.rodata*) *(.srodata*) }
  .data : {
    PROVIDE(__global_pointer$ = . + 0x800);
    *(.data*) *(.sdata*)
  }
  .tdata ALIGN(16) : {
    __tls_start = .;
    *(.tdata*)
    __tdata_end = .;
  }
  .tbss ALIGN(8) (NOLOAD) : {
    *(.tbss*)
    __tls_end = .;
  }
  . = ADDR(.tbss) + SIZEOF(.tbss);
  .bss ALIGN(8) (NOLOAD) : {
    __bss_start = .;
    *(.bss*) *(.sbss*) *(COMMON)
    . = ALIGN(8);
    __bss_end = .;
  }
  ASSERT(__tls_end - __tls_start <= 64, "Goban task fixture TLS exceeds explicit 64-byte task TLS slots")
  ASSERT(. < 0x81000000, "Goban task fixture data exceeds its declared 16-MiB image region")
  . = ALIGN(16);
  PROVIDE(_end = .);
  PROVIDE(end = .);
}
