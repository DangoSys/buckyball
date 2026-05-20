SECTIONS
{
    ROM_BASE = 0x10000;

    . = ROM_BASE;
    .text.hang : { *(.text.hang) }
    . = ROM_BASE + 0x40;
    .text.start : { *(.text.start) }

    /*
     * _dtb symbol points to the byte immediately after the bootrom binary.
     * Rocket Chip's BootROM scala code appends the DTB at this exact offset
     * by concatenating dtb.contents to romdata.array(). This ensures the
     * bootrom's `la a1, _dtb` computes the correct DTB address at runtime.
     *
     * No alignment here — the bootrom binary ends exactly at _dtb, so the
     * appended DTB starts at this byte. Any extra padding would cause a
     * mismatch between the computed address and the actual DTB location.
     */
    _dtb = .;
}
