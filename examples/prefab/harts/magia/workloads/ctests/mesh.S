.section .magia,"a",@progbits
.incbin "mesh.bin"

.section .rodata.magia,"a",@progbits
.balign 8
.global _binary_mesh_bin_start
.global _binary_mesh_bin_end
_binary_mesh_bin_start:
.incbin "mesh.bin", 128, 64
_binary_mesh_bin_end:
