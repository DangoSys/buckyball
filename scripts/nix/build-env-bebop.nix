{ pkgs, bebopPkgs ? {} }:

{
  # Rust toolchain
  rustc = pkgs.rustc;
  cargo = pkgs.cargo;
  rustfmt = pkgs.rustfmt;
  clippy = pkgs.clippy;

  # Bebop packages (from bebop flake)
  bebop = bebopPkgs.bebop or null;
  bebopHost = bebopPkgs.host or null;
  spike = bebopPkgs.spike or null;
  gem5 = bebopPkgs.gem5 or null;
}
