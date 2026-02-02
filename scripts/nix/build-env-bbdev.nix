{ pkgs }:

{
  # Node.js environment
  nodejs = pkgs.nodejs_20;
  npm = pkgs.nodejs_20;

  # Build tools (for compiling native modules)
  gcc = pkgs.gcc;
  gnumake = pkgs.gnumake;
  pkg-config = pkgs.pkg-config;

  # System dependencies
  systemd = pkgs.systemd;
}
