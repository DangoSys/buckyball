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
        # nix build
        packages.bbdev = pkgs.bbdevCli;
        packages.default = pkgs.bbdevCli;

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Node.js runtime (Motia is a Node.js framework)
            nodejs_22
            nodePackages.pnpm

            # Python with project dependencies (managed by Nix, not requirements.txt)
            bbdevPythonPkgs

            # System tools
            curl
            git
          ];

          shellHook = ''
            export PATH="$PWD:$PATH"

            echo "bbdev dev environment ready"
            echo "  Node.js: $(node --version)"
            echo "  Python:  $(python3 --version)"
            echo "  pnpm:    $(pnpm --version)"
            echo "  bbdev:   $(which bbdev)"
            echo ""
            echo "Quick start:"
            echo "  pnpm install     # install node deps + motia install"
            echo "  pnpm dev         # start motia dev server"
          '';
        };
      }
    );
}
