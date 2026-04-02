{
  description = "bbdev - Agent-friendly developer toolchain backend powered by Motia";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import ./nix/overlay.nix) ];
        pkgs = import nixpkgs { inherit system overlays; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python with system-level dependencies (managed by Nix)
            bbdevPythonPkgs

            # System tools
            curl
            git
          ];

          shellHook = ''
            export PATH="$PWD:$PATH"

            # Create venv for pip-managed packages (motia, iii-sdk)
            VENV_DIR="$PWD/api/.venv"
            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating Python venv..."
              python3 -m venv "$VENV_DIR" --system-site-packages
              "$VENV_DIR/bin/pip" install -q motia[otel]==1.0.0rc17 iii-sdk==0.2.0
            fi
            export PATH="$VENV_DIR/bin:$PATH"

            echo "bbdev dev environment ready"
            echo "  Python:  $(python3 --version)"
            echo "  bbdev:   $(which bbdev)"
          '';
        };
      }
    );
}
