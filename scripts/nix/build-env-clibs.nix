{ pkgs }:

{
  # C libraries needed by Verilator build
  zlib-dev = pkgs.zlib.dev;
  zlib = pkgs.zlib;
  # C libraries needed by bdb debugger
  readline-dev = pkgs.readline.dev;
  readline = pkgs.readline;
  # buddy DIP imgcodecs (grfmt_jpeg.h)
  jpeg-dev = pkgs.libjpeg.dev;
  jpeg = pkgs.libjpeg;
  # buddy DIP imgcodecs (grfmt_png.h)
  png-dev = pkgs.libpng.dev;
  png = pkgs.libpng;
}
