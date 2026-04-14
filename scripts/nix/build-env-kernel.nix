{ pkgs }:

{
  # Host tools for RISC-V kernel/rootfs assembly
  # mke2fs: pack rootfs directory into ext4 image
  e2fsprogs = pkgs.e2fsprogs;
}
