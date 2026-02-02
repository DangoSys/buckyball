{
  description = "Development environment for Buckyball with Verilator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSystem = f: nixpkgs.lib.genAttrs systems (system: f system);
    in
    {
      devShells = forEachSystem (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            packages = [
              # Fast and robust (System)Verilog simulator/compiler and linter
              pkgs.verilator

              # RISC-V embedded toolchain (bare metal)
              pkgs.pkgsCross.riscv64-embedded.buildPackages.gcc

              # RISC-V Linux toolchain
              pkgs.pkgsCross.riscv64.buildPackages.gcc

              # Build tool for Scala, Java and more
              pkgs.mill
            ];

            shellHook = ''
              echo "Verilator: $(verilator --version 2>&1 | head -1)"
              echo "RISC-V Embedded GCC: $(riscv64-none-elf-gcc --version 2>&1 | head -1)"
              echo "RISC-V Linux GCC: $(riscv64-unknown-linux-gnu-gcc --version 2>&1 | head -1)"
              echo "Mill: cd arch && $(mill --version 2>&1 | head -1) && cd .."
            '';
          };
        }
      );
    };
}
