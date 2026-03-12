/* goban.ld — baremetal linker script for Goban multi-core workloads (BBSimHarness) */
OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS {
    . = 0x80000000;

    .text : {
        *(.text.init)
        *(.text.startup .text.startup.*)
        *(.text .text.*)
        *(.gnu.linkonce.t.*)
    }

    .rodata : {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
        *(.gnu.linkonce.r.*)
    }

    .data : {
        *(.data .data.*)
        *(.sdata .sdata.*)
        *(.gnu.linkonce.d.*)
        *(.gnu.linkonce.s.*)
    }

    .bss (NOLOAD) : {
        PROVIDE(__bss_start = .);
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(.gnu.linkonce.b.*)
        *(.gnu.linkonce.sb.*)
        *(COMMON)
        PROVIDE(__bss_end = .);
    }

    . = ALIGN(16);
    PROVIDE(end = .);
    PROVIDE(_end = .);
}
