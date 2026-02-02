{ pkgs }:

{
  # Rust toolchain
  rustc = pkgs.rustc;
  cargo = pkgs.cargo;
  rustfmt = pkgs.rustfmt;
  clippy = pkgs.clippy;
}
