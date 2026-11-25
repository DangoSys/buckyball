/* bbsim.ld — baremetal linker script for BBSimHarness (DRAM at 0x8000_0000) */
/* Bootrom jumps directly to 0x80000000 (ELF entry), _start must be first.  */
OUTPUT_ARCH("riscv")
ENTRY(_start)

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
        *(.srodata .srodata.*)
        *(.gnu.linkonce.r.*)
    }

    .data : {
        *(.data .data.*)
        *(.sdata .sdata.*)
        *(.gnu.linkonce.d.*)
        *(.gnu.linkonce.s.*)
        PROVIDE(_edata = .);
    }

    .tdata ALIGN(16) : {
        PROVIDE(__tls_start = .);
        *(.tdata .tdata.* .gnu.linkonce.td.*)
        PROVIDE(__tdata_end = .);
    }

    .tbss ALIGN(8) (NOLOAD) : {
        PROVIDE(__bss_start = .);
        *(.tbss .tbss.* .gnu.linkonce.tb.*)
        . = ALIGN(8);
        PROVIDE(__tls_end = .);
    }
    . = ADDR(.tbss) + SIZEOF(.tbss);

    /* All zero-init sections: crt0.S clears __bss_start .. __bss_end,
       so heap (at _end) starts after ALL bss including sbss. */
    .bss (NOLOAD) : {
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(.gnu.linkonce.b.*)
        *(.gnu.linkonce.sb.*)
        *(COMMON)
        . = ALIGN(8);
        PROVIDE(__bss_end = .);
    }

    .noinit (NOLOAD) : {
        *(.noinit .noinit.*)
    }

    . = ALIGN(16);
    PROVIDE(end = .);
    PROVIDE(_end = .);
}
