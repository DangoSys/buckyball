{
  description = "Development environment for Buckyball with Verilator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }@inputs:
    let
      overlay = import ./scripts/nix/overlay.nix;
    in
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs { overlays = [ overlay ]; inherit system; };
        in
        {
          legacyPackages = pkgs;

          # nix build
          packages.default = pkgs.buildEnv {
            name = "buckyball-environment";
            paths = with pkgs; [
              # Chipyard tools
              chipyard.verilator
              chipyard.riscv-embedded-gcc
              chipyard.riscv-linux-gcc
              chipyard.mill

              # Bebop tools
              bebop.rustc
              bebop.cargo
              bebop.rustfmt
              bebop.clippy
            ];
          };

          # nix develop
          devShells.default = pkgs.mkShell {
            shellHook = ''
              if [ -d "$PWD/result/bin" ]; then
                export PATH="$PWD/result/bin:$PATH"
                echo "Activated environment"
              else
                echo "Warning: result/bin not found. Run 'nix build' first."
              fi
              echo "=========================================="
              echo "Development environment loaded:"
              echo "Verilator: $(verilator --version 2>&1 | head -1)"
              echo "RISC-V Embedded GCC: $(riscv64-none-elf-gcc --version 2>&1 | head -1)"
              echo "RISC-V Linux GCC: $(riscv64-unknown-linux-gnu-gcc --version 2>&1 | head -1)"
              echo "Mill: $(mill --version 2>&1 | head -1)"
              echo "=========================================="
            '';
          };
        }
      ) // {
      inherit inputs;
      overlays.default = overlay;
    };
}
