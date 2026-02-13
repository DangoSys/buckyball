{ pkgs }:

{
  # Node.js environment (for bbdev Motia backend)
  nodejs = pkgs.nodejs_22;
  pnpm = pkgs.nodePackages.pnpm;

  # Build tools (for compiling native modules)
  gcc = pkgs.gcc;
  gnumake = pkgs.gnumake;
  pkg-config = pkgs.pkg-config;
}
