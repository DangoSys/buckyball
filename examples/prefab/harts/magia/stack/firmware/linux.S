# The firmware of the tiles in a Linux program of Loom, which copies it to
# the start of L2 (runtime/mesh/linux.c).
.section .rodata
.global magia_firmware
.global magia_firmware_end
magia_firmware:
.incbin "mesh.bin"
magia_firmware_end:
