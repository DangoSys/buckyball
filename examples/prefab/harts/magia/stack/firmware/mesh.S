# The firmware of the tiles in a program of Loom: the section the program's
# linker script puts at the start of L2, where a launch boots the tiles.
.section .magia,"a",@progbits
.incbin "mesh.bin"
