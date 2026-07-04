{ pkgs }:

{
  # Rust toolchain (waveform-mcp, etc.)
  rustc = pkgs.rustc;
  cargo = pkgs.cargo;
  cargoNextest = pkgs.cargo-nextest;
  rustfmt = pkgs.rustfmt;
  clippy = pkgs.clippy;
}
