/* bbsim.ld — baremetal linker script for BBSimHarness (DRAM at 0x8000_0000) */
/* Bootrom jumps directly to 0x80000000 (ELF entry), no _start needed.      */
OUTPUT_ARCH("riscv")
ENTRY(main)

SECTIONS {
    . = 0x80000000;

    .text : {
        *(.text.init)           /* crt0.S _start must be first */
        *(.text.startup .text.startup.*)
        *(.text .text.*)
        *(.gnu.linkonce.t.*)
    }

    .rodata : {
        *(.rodata .rodata.*)
        *(.gnu.linkonce.r.*)
    }

    .data : {
        *(.data .data.*)
        *(.gnu.linkonce.d.*)
        PROVIDE(_edata = .);
    }

    .bss (NOLOAD) : {
        PROVIDE(__bss_start = .);
        *(.bss .bss.*)
        *(.gnu.linkonce.b.*)
        *(COMMON)
        PROVIDE(__bss_end = .);
    }

    . = ALIGN(16);
    PROVIDE(end = .);
    PROVIDE(_end = .);
}
