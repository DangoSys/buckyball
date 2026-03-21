{ pkgs }:

{
  # Rust toolchain (waveform-mcp, etc.)
  rustc = pkgs.rustc;
  cargo = pkgs.cargo;
  rustfmt = pkgs.rustfmt;
  clippy = pkgs.clippy;
}
