{ pkgs }:

{
  # Rust toolchain (for building bebop)
  rustc = pkgs.rustc;
  cargo = pkgs.cargo;
  rustfmt = pkgs.rustfmt;
  clippy = pkgs.clippy;
}
