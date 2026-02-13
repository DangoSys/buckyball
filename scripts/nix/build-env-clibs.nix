{ pkgs }:

{
  # C libraries needed by Verilator build
  zlib-dev = pkgs.zlib.dev;
  zlib = pkgs.zlib;
  # C libraries needed by bdb debugger
  readline-dev = pkgs.readline.dev;
  readline = pkgs.readline;
}
