/* device.ld — baremetal linker script of programs on the MAGIA device (DRAM at
   0x8000_0000). The firmware of the tiles lies where a launch boots them, at
   0xCC00_0000, with 16 MiB for its job and data; the program is below it,
   the heap above. The parameters of a model go on either side
   (.parameters, .parameters.high): the mesh reaches any of the DRAM through
   its windows. */
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

    /* Parameters of a model below the firmware. */
    .parameters ALIGN(64) : {
        *(.parameters)
    }
    ASSERT(. <= 0xCC000000, "the program runs into the firmware of the tiles")

    /* The firmware of the tiles (the hart's mesh.o): instructions from
       0xCC00_0000, the job of a launch at 0xCC00_F000, its data from
       0xCC01_0000. */
    .magia 0xCC000000 : {
        KEEP(*(.magia))
    }

    /* Above the firmware: more parameters, then the heap. The address is the
       location counter's, not only the section's: a program without
       .parameters.high would start its heap in the data of the firmware. */
    . = 0xCD000000;
    .parameters.high : {
        *(.parameters.high)
    }

    . = ALIGN(16);
    PROVIDE(end = .);
    PROVIDE(_end = .);
}
