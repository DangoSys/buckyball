{
  description = "Development environment for Buckyball with Verilator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }@inputs:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          overlay = import ./scripts/nix/overlay.nix;
          pkgs = import nixpkgs { overlays = [ overlay ]; inherit system; };
        in
        {
          legacyPackages = pkgs;

          # nix build
          packages.default = pkgs.buildEnv {
            name = "buckyball-environment";
            paths = with pkgs; [
              tools.verilator
              tools.dramsim2
              tools.ccache
              tools.lld
              tools.cmake
              tools.java
              tools.dtc
              tools.spike
              tools.yosys
              tools.opensta
              tools.lcov

              # RISC-V toolchain
              riscv.riscv-embedded-gcc
              riscv.riscv-linux-gcc

              # python environment
              python.python3Packages
              pkgs."pre-commit"
              pkgs.clang-tools  # clang-format for pre-commit (language: system)

              # Rust toolchain
              rustTools.rustc
              rustTools.cargo
              rustTools.rustfmt
              rustTools.clippy

              # bbdev dependencies
              bbdev.iii
              bbdev.uv
              bbdev.allure
              bbdev.gcc
              bbdev.gnumake
              bbdev.pkg-config

              # Kernel build tools (RISC-V kernel + rootfs for Pegasus)
              kernel.e2fsprogs

              # C libraries (headers + link libs)
              clibs.zlib-dev
              clibs.zlib
              clibs.readline-dev
              clibs.readline
              clibs.jpeg-dev
              clibs.jpeg
              clibs.png-dev
              clibs.png

              # Scala tools
              scala.mill
              scala.sbt
              scala.scalafmt
              scala.coursier

              # Documentation tools
              doc.mdbook
              doc.mdbook-linkcheck
              doc.mdbook-pdf
              doc.mdbook-toc
              doc.mdbook-mermaid

              # System utilities
              systemTools.rsync
              systemTools.nodejs

              git
            ];
          };

          # nix develop
          devShells.default = pkgs.mkShell {
            buildInputs = with pkgs; [
              git
              clibs.zlib-dev
              clibs.zlib
              clibs.readline-dev
              clibs.readline
              clibs.jpeg-dev
              clibs.jpeg
              clibs.png-dev
              clibs.png
            ];
            shellHook = ''
              if [ -d "$PWD/result/bin" ]; then
                export PATH="$PWD/result/bin:$PATH"
              else
                echo "Warning: result/bin not found. Run 'nix build' first." >&2
              fi

              source "$PWD/sourceme.sh"

              # Verilator build acceleration: ccache via OBJCACHE
              export OBJCACHE=ccache

              if [ -z "$NIX_QUIET" ]; then
                echo "================= Buckyball Environment Activated ========================="
                echo "Development environment loaded:"
                echo "Verilator: $(verilator --version 2>&1 | head -1)"
                echo "RISC-V Embedded GCC: $(riscv64-unknown-elf-gcc --version 2>&1 | head -1)"
                echo "RISC-V Linux GCC: $(riscv64-unknown-linux-gnu-gcc --version 2>&1 | head -1)"
                echo "Mill: $(mill --version 2>&1 | head -1)"
                echo "Cargo: $(cargo --version 2>&1 | head -1)"
                echo "npm: $(npm --version 2>&1 | head -1)"
                echo "bbdev: $(which bbdev)"
                echo "RISCV: $RISCV"
                echo "Yosys: $(yosys --version 2>&1 | head -1)"
                echo "OpenSTA: $(sta -version 2>&1 | head -1)"
                echo "Buddy MLIR: $(which buddy-opt)"
                echo "==========================================================================="
              fi
            '';
          };
        }
      );
}
