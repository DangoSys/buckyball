{ pkgs }:

let
  # libdwarf 0.x (legacy) for FireSim TracerV compatibility
  # FireSim's tracerv code uses the old libdwarf API (pre-2.0 rewrite)
  # Corresponds to libdwarf-dev==0.0.0.20190110 from FireSim conda config
  libdwarf-legacy = pkgs.stdenv.mkDerivation rec {
    pname = "libdwarf-legacy";
    version = "20190112";

    src = pkgs.fetchFromGitHub {
      owner = "davea42";
      repo = "libdwarf-code";
      # Commit from 2019-01-13 (version 20190112, close to FireSim's 20190110)
      rev = "6af2758f8c24e7fa8302befa0fb82d9c31d77889";
      sha256 = "sha256-/GCHGcMc6Mf7EQ1G5KS525KPgj5iqU7ksyfIJ6zcxyM=";
    };

    # Don't use autoreconfHook - the repo already has a working configure script
    buildInputs = with pkgs; [ elfutils zlib ];

    configureFlags = [
      "--enable-shared"
      "--disable-static"
    ];

    # Remove dwarf.h to avoid conflict with elfutils' dwarf.h in buildEnv
    postInstall = ''
      rm -f $out/include/dwarf.h
    '';

    meta = with pkgs.lib; {
      description = "Library for DWARF debug symbols (legacy 0.x for FireSim)";
      homepage = "https://www.prevanders.net/dwarf.html";
      platforms = platforms.linux;
    };
  };
in
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
  # ELF parsing for kernel loading
  elfutils-dev = pkgs.elfutils.dev;
  elfutils = pkgs.elfutils.out;  # Use .out for libelf.so (not .bin which is default)
  # GMP needed by FireSim driver (peek_poke.h includes <gmp.h>)
  gmp-dev = pkgs.gmp.dev;
  gmp = pkgs.gmp;
  # libdwarf needed by FireSim TracerV bridge.
  # Use legacy 0.x version (2019) for API compatibility with firechip tracerv code.
  libdwarf-dev = libdwarf-legacy;
  libdwarf = libdwarf-legacy;
}
