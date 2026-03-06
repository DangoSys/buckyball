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
              tools.yosys
              tools.opensta

              # RISC-V toolchain
              riscv.riscv-embedded-gcc
              riscv.riscv-linux-gcc

              # python environment
              python.python3Packages
              pkgs."pre-commit"
              pkgs.clang-tools  # clang-format for pre-commit (language: system)

              # Bebop dependencies (rust toolchain)
              bebop.rustc
              bebop.cargo
              bebop.rustfmt
              bebop.clippy

              # bbdev dependencies
              bbdev.nodejs
              bbdev.pnpm
              bbdev.uv
              bbdev.allure
              bbdev.gcc
              bbdev.gnumake
              bbdev.pkg-config

              # C libraries (headers + link libs)
              clibs.zlib-dev
              clibs.zlib
              clibs.readline-dev
              clibs.readline

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

              source "$PWD/sourceme.sh"

              # Verilator build acceleration: ccache via OBJCACHE
              export OBJCACHE=ccache

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
              echo "Yosys: $(yosys --version 2>&1 | head -1)"
              echo "OpenSTA: $(sta -version 2>&1 | head -1)"
              echo "==========================================================================="
            '';
          };
        }
      ) // {
      inherit inputs;
    };
}
