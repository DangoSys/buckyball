{
  description = "Pegasus FPGA simulation framework for AU280 (Buckyball project)";

  inputs = {
    nixpkgs.url    = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        # nix develop — C++ driver build + Vivado environment
        devShells.default = pkgs.mkShell {
          name = "pegasus-dev";
          packages = with pkgs; [
            cmake
            gcc
            gnumake
            pkg-config
            libelf
            python3
            python3Packages.pyelftools
            git
          ];
          shellHook = ''
            for ver in 2024.1 2023.2 2022.2; do
              if [ -f /tools/Xilinx/Vivado/$ver/settings64.sh ]; then
                source /tools/Xilinx/Vivado/$ver/settings64.sh
                echo "Vivado $ver loaded."
                break
              fi
            done
            echo "Pegasus dev shell ready."
            echo "  Build driver:  cmake -B driver/build driver && cmake --build driver/build"
            echo "  Install XDMA:  nix run .#install-xdma"
          '';
        };

        # nix build .#install-xdma  — produces result/bin/install-xdma
        # nix run   .#install-xdma  — build + run directly
        #
        # Clones XDMA sources and compiles against the *running* kernel at
        # execution time, so the .ko matches whatever kernel the server has.
        packages.install-xdma = pkgs.writeShellApplication {
          name = "install-xdma";
          runtimeInputs = with pkgs; [ git gnumake gcc kmod ];
          text = ''
            KERNEL=$(uname -r)
            KBUILD=/lib/modules/$KERNEL/build

            if [ ! -d "$KBUILD" ]; then
              echo "ERROR: Kernel headers not found at $KBUILD"
              echo "       Run: sudo apt install linux-headers-$KERNEL"
              exit 1
            fi

            TMPDIR=$(mktemp -d)
            trap "rm -rf $TMPDIR" EXIT

            echo "==> Cloning XDMA driver source..."
            git clone https://github.com/joonho3020/dma_ip_drivers "$TMPDIR/src"
            git -C "$TMPDIR/src" checkout ubuntu-24-xdma

            echo "==> Building against kernel $KERNEL..."
            make -C "$TMPDIR/src/XDMA/linux-kernel/xdma" KERNELDIR="$KBUILD"

            echo "==> Installing (requires sudo)..."
            sudo make -C "$TMPDIR/src/XDMA/linux-kernel/xdma" KERNELDIR="$KBUILD" install
            sudo modprobe xdma

            echo "==> Verifying..."
            if ls /dev/xdma* 2>/dev/null; then
              echo "XDMA installed successfully!"
            else
              echo "Module loaded. No /dev/xdma* yet — connect AU280 and re-run: sudo modprobe xdma"
            fi
          '';
        };

        apps.install-xdma = {
          type    = "app";
          program = "${pkgs.lib.getExe self.packages.${system}.install-xdma}";
        };
      }
    );
}
