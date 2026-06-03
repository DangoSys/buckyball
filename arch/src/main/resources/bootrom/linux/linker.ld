SECTIONS
{
    ROM_BASE = 0x10000;

    . = ROM_BASE;
    .text.hang : { *(.text.hang) }
    . = ROM_BASE + 0x40;
    .text.start : { *(.text.start) }

    /*
     * Rocket Chip appends the DTB immediately after this image. Keep _dtb at
     * the image end so the bootrom address and appended DTB offset match.
     */
    . = ALIGN(8);
    _dtb = .;
}
