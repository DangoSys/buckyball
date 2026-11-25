/* A thread program alone on its virtual environment. The program image is
 * at the start of the ideal memory (MEMORY_BASE): a header of the code bytes
 * and the TLS image bytes, then the code (run at 0, in the code store) and
 * the TLS image (run at TLS_BASE: data, constants and zeroed variables). The
 * environment loads both through the TSS and starts the task at 0 with the
 * stack at the top of the TLS. CODE_BYTES, TLS_BASE, TLS_BYTES and
 * MEMORY_BASE are the thread's (--defsym). */
OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS {
  .image MEMORY_BASE : { QUAD(SIZEOF(.text)) QUAD(SIZEOF(.tls)) }
  .text 0 : AT(MEMORY_BASE + 16) {
    KEEP(*(.text.start)) *(.text .text.*) . = ALIGN(16);
  }
  .tls TLS_BASE : AT(MEMORY_BASE + 16 + SIZEOF(.text)) {
    *(.rodata .rodata.* .srodata .srodata.*)
    *(.data .data.* .sdata .sdata.*)
    *(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(16);
  }
  ASSERT(SIZEOF(.text) <= CODE_BYTES, "Thread program exceeds the code store")
  ASSERT(SIZEOF(.tls) + 4096 <= TLS_BYTES, "Thread TLS image leaves no 4 KiB stack")
  /DISCARD/ : { *(.comment .eh_frame .eh_frame.* .riscv.attributes) }
}
