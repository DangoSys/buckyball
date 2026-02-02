{ pkgs }:

{
  # Fast and robust (System)Verilog simulator/compiler and linter
  verilator = pkgs.verilator;

  # RISC-V embedded toolchain (bare metal)
  riscv-embedded-gcc = pkgs.pkgsCross.riscv64-embedded.buildPackages.gcc;

  # RISC-V Linux toolchain
  riscv-linux-gcc = pkgs.pkgsCross.riscv64.buildPackages.gcc;

  # Build tool for Scala, Java and more
  mill = pkgs.mill;
}
