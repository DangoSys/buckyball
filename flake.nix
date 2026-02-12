{
  description = "Development environment for Buckyball with Verilator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # flake-utils.url = "github:numtide/flake-utils";
    bebop = {
      url = "github:DangoSys/bebop/5f65c8f264b176845924c1ec17935ae4c3e103d7";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, bebop }@inputs:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          bebopPkgs = bebop.packages.${system};
          overlay = import ./scripts/nix/overlay.nix { inherit bebopPkgs; };
          pkgs = import nixpkgs { overlays = [ overlay ]; inherit system; };
        in
        {
          legacyPackages = pkgs;

          # nix build
          packages.default = pkgs.buildEnv {
            name = "buckyball-environment";
            paths = with pkgs; [
              tools.verilator

              # RISC-V toolchain
              riscv.riscv-embedded-gcc
              riscv.riscv-linux-gcc

              # python environment
              python.python3Packages

              # Bebop tools
              pkgs.bebop.rustc
              pkgs.bebop.cargo
              pkgs.bebop.rustfmt
              pkgs.bebop.clippy
              pkgs.bebop.bebop
              pkgs.bebop.bebopHost
              pkgs.bebop.spike
              pkgs.bebop.gem5

              # Workflow dev tools
              bbdev.nodejs
              bbdev.gcc
              bbdev.gnumake
              bbdev.pkg-config

              # Scala tools
              scala.mill
              scala.scalafmt
              scala.coursier

              # Documentation tools
              doc.mdbook
              doc.mdbook-linkcheck
              doc.mdbook-pdf
              doc.mdbook-toc
              doc.mdbook-mermaid
            ];
          };

          # nix develop
          devShells.default = pkgs.mkShell {
            shellHook = ''
              if [ -d "$PWD/result/bin" ]; then
                export PATH="$PWD/result/bin:$PATH"
                echo "================= Buckyball Environment Activated ========================="
              else
                echo "Warning: result/bin not found. Run 'nix build' first."
              fi

              export BUDDY_MLIR_BUILD_DIR="$PWD/compiler/build"
              export LLVM_MLIR_BUILD_DIR="$PWD/compiler/llvm/build"
              export PYTHONPATH="$PWD/compiler/llvm/build/tools/mlir/python_packages/mlir_core:$PWD/compiler/build/python_packages:$PYTHONPATH"
              export PATH="$PWD/workflow:$PATH"
              export RISCV="$PWD/result"
              export PATH="$PWD/thirdparty/libgloss/install/lib:$PATH"

              echo "Development environment loaded:"
              echo "Verilator: $(verilator --version 2>&1 | head -1)"
              echo "RISC-V Embedded GCC: $(riscv64-unknown-elf-gcc --version 2>&1 | head -1)"
              echo "RISC-V Linux GCC: $(riscv64-unknown-linux-gnu-gcc --version 2>&1 | head -1)"
              echo "Bebop: $(which bebop)"
              echo "Spike: $(which spike)"
              echo "Bebop Gem5: $(which gem5.opt)"
              echo "Mill: $(mill --version 2>&1 | head -1)"
              echo "Cargo: $(cargo --version 2>&1 | head -1)"
              echo "npm: $(npm --version 2>&1 | head -1)"
              echo "bbdev: $(which bbdev)"
              echo "RISCV: $RISCV"
              echo "==========================================================================="
            '';
          };
        }
      ) // {
      inherit inputs;
    };
}
