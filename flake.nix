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
          pkgs = import nixpkgs {
            overlays = [ overlay ];
            inherit system;
            config.allowUnfree = true;
            config.permittedInsecurePackages = [
              "openjdk-8u502-b07"
            ];
          };

          defaultShell = pkgs.mkShell {
            # this is needed to build LLVM/libc++, disable the nix injected hardening
            hardeningDisable = [ "libcxxhardeningfast" ];

            buildInputs = with pkgs; [
	      python.python3Packages
              clibs.zlib-dev
              clibs.zlib
              clibs.lz4-dev
              clibs.lz4
              clibs.readline-dev
              clibs.readline
              clibs.jpeg-dev
              clibs.jpeg
              clibs.png-dev
              clibs.png
              clibs.elfutils-dev
              clibs.elfutils
              clibs.gmp-dev
              clibs.gmp
              clibs.libdwarf-dev
              clibs.libdwarf

              compiler.flatbuffers
              compiler.numactl

              rustTools.cargoNextest

              # protoc for bbdev config --install (chip.proto -> chip.pb)
              pkgs.protobuf

              pkgs.xorg-server
              pkgs.jdk8
            ];
            shellHook = ''
              # Must run with cwd at the git checkout. Store copies from toString ./.
              # are unusable for development paths. MCP launcher cds to repo root first.
              if [ ! -f "$PWD/sourceme.sh" ]; then
                echo "ERROR: nix develop cwd must be the buckyball repo root (missing $PWD/sourceme.sh)." >&2
                echo "Use: cd <repo> && nix develop   or   scripts/claude/run_mcp_server.sh" >&2
                return 1 2>/dev/null || exit 1
              fi
              BB_ROOT="$PWD"
              export DSH_HOME="$BB_ROOT/.dsh"
              if [ -d "$BB_ROOT/result/bin" ]; then
                export PATH="$BB_ROOT/result/bin:$PATH"
              else
                echo "Warning: result/bin not found. Run 'nix build' first." >&2
              fi

              source "$BB_ROOT/sourceme.sh"

              # Verilator build acceleration: ccache via OBJCACHE
              export OBJCACHE=ccache

              export CUDA_HOME="${pkgs.cuda.cudatoolkit}"
              export CPATH="''${CUDA_HOME}/include''${CPATH:+:$CPATH}"

              export CC="${pkgs.systemTools.clang}/bin/clang"
              export CXX="${pkgs.systemTools.clang}/bin/clang++"

              export JAVA_HOME="${pkgs.jdk17}"
              # used by smic180
              export S018SP_JAVA="${pkgs.jdk8}/bin/java"
              export PATH="$JAVA_HOME/bin:${pkgs.xorg-server}/bin:$PATH"

              # Banner must go to stderr. stdout is reserved for MCP stdio / machine parsers.
              if [ -z "$NIX_QUIET" ]; then
                echo "================= Buckyball Environment Activated =========================" >&2
                echo "Development environment loaded:" >&2
                echo "Verilator: $(verilator --version 2>&1 | head -1)" >&2
                echo "RISC-V Embedded GCC: $(riscv64-unknown-elf-gcc --version 2>&1 | head -1)" >&2
                echo "RISC-V Linux GCC: $(riscv64-unknown-linux-gnu-gcc --version 2>&1 | head -1)" >&2
                echo "Mill: $(mill --version 2>&1 | head -1)" >&2
                echo "Cargo: $(cargo --version 2>&1 | head -1)" >&2
                echo "pnpm: $(pnpm --version 2>&1 | head -1)" >&2
                echo "CXX: $CXX" >&2
                echo "bbdev: $(which bbdev)" >&2
                echo "RISCV: $RISCV" >&2
                echo "Yosys: $(yosys --version 2>&1 | head -1)" >&2
                echo "OpenSTA: $(sta -version 2>&1 | head -1)" >&2
                # echo "Buddy MLIR: $(which buddy-opt)" >&2
                echo "===========================================================================" >&2
              fi
            '';
          };
        in
        {
          legacyPackages = pkgs;

          packages.eda = pkgs.buildEnv {
            name = "buckyball-eda-environment";
            paths = with pkgs.eda; [
              yosys
              opensta
              openroad
              magic
              netgen
              klayout
              sky130
            ];
          };

          # nix build
          packages.default = pkgs.buildEnv {
            name = "buckyball-environment";
            paths = with pkgs; [
              tools.verilator
              tools.dramsim3
              tools.ccache
              tools.lld
              tools.cmake
              tools.ninja
              tools.java
              tools.dtc
              tools.spike
              tools.yosys
              tools.opensta
              tools.lcov
              tools.verible

              # RISC-V toolchain
              riscv.riscv-embedded-gcc
              riscv.riscv-linux-gcc

              # python environment
              python.python3Packages
              pkgs."pre-commit"
              # clang-format for pre-commit (language: system)
              pkgs.clang-tools

              # Rust toolchain
              rustTools.rustc
              rustTools.cargo
              rustTools.cargoNextest
              rustTools.rustfmt
              rustTools.clippy
              rustTools.rustAnalyzer

              # bbdev dependencies
              bbdev.iii
              bbdev.uv
              bbdev.gcc
              bbdev.gnumake
              bbdev.pkg-config

              # Kernel build tools (RISC-V kernel + rootfs)
              kernel.e2fsprogs

              # C libraries (headers + link libs)
              clibs.zlib-dev
              clibs.zlib
              clibs.lz4-dev
              clibs.lz4
              clibs.readline-dev
              clibs.readline
              clibs.jpeg-dev
              clibs.jpeg
              clibs.png-dev
              clibs.png
              clibs.elfutils-dev
              clibs.elfutils
              clibs.gmp-dev
              clibs.gmp
              clibs.libdwarf-dev
              clibs.libdwarf

              # Compiler tools
              compiler.flatbuffers
              compiler.numactl

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
              systemTools.git
              systemTools.pnpm

              # CUDA toolkit (12.8, matches host driver) + host g++-13
              cuda.cudatoolkit
              cuda.gcc13
            ];
          };

          # nix develop
          devShells.default = defaultShell;

          # default + EDA tools / sky130
          devShells.full = pkgs.mkShell {
            inputsFrom = [ defaultShell ];
            packages = with pkgs.eda; [
              yosys
              opensta
              openroad
              magic
              netgen
              klayout
              sky130
            ];
            shellHook = ''
              export SKY130_ROOT="${pkgs.eda.sky130}"
              echo "Full environment activated (default + EDA)" >&2
              echo "  libraries: $SKY130_ROOT" >&2
            '';
          };
        }
      );
}
