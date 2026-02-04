{ pkgs }:

let
  # Build newlib-nano with size optimization flags
  newlib-nano = pkgs.pkgsCross.riscv64-embedded.newlib.overrideAttrs (oldAttrs: {
    pname = "newlib-nano";
    configureFlags = oldAttrs.configureFlags or [] ++ [
      "--enable-newlib-nano-malloc"
      "--enable-newlib-nano-formatted-io"
      "--enable-newlib-reent-small"
      "--disable-newlib-fvwrite-in-streamio"
      "--disable-newlib-fseek-optimization"
      "--disable-newlib-wide-orient"
      "--disable-newlib-unbuf-stream-opt"
      "--enable-lite-exit"
      "--enable-newlib-global-atexit"
    ];
    CFLAGS_FOR_TARGET = "-Os -ffunction-sections -fdata-sections -mcmodel=medany";
  });

  # Create a custom cross system with newlib-nano
  riscv64EmbeddedWithNano = pkgs.pkgsCross.riscv64-embedded.stdenv.targetPlatform // {
    libc = "newlib-nano";
  };

  # Build the toolchain with the custom platform
  pkgsCrossWithNano = import pkgs.path {
    inherit (pkgs) system;
    crossSystem = riscv64EmbeddedWithNano;
    overlays = [
      (self: super: {
        newlib = newlib-nano;
      })
    ];
  };

  # Build libgloss-htif library from upstream
  libgloss-htif = pkgs.stdenv.mkDerivation {
    pname = "libgloss-htif";
    version = "0.1";

    src = pkgs.fetchFromGitHub {
      owner = "ucb-bar";
      repo = "libgloss-htif";
      rev = "master";
      sha256 = "sha256-FXuN1xK5133QqoHI4EG7mvhk7K8J6//ar7Y1+IUPER0=";
    };

    nativeBuildInputs = [ pkgsCrossWithNano.buildPackages.gcc ];

    configureFlags = [
      "--host=riscv64-unknown-elf"
    ];

    preConfigure = ''
      export CC=riscv64-none-elf-gcc
      export AR=riscv64-none-elf-ar
      export RANLIB=riscv64-none-elf-ranlib
    '';

    enableParallelBuilding = true;

    installPhase = ''
      mkdir -p $out/riscv64-unknown-elf/lib

      # Install library and specs files manually to bypass the Makefile check
      install -m 644 libgloss_htif.a $out/riscv64-unknown-elf/lib/
      install -m 644 util/htif.specs $out/riscv64-unknown-elf/lib/
      install -m 644 util/htif_nano.specs $out/riscv64-unknown-elf/lib/
      install -m 644 util/htif_wrap.specs $out/riscv64-unknown-elf/lib/
      install -m 644 util/htif_argv.specs $out/riscv64-unknown-elf/lib/
      install -m 644 util/htif.ld $out/riscv64-unknown-elf/lib/
    '';
  };
in

{
  # Fast and robust (System)Verilog simulator/compiler and linter
  verilator = pkgs.verilator;

  # RISC-V embedded toolchain (bare metal), with riscv64-unknown-elf-* symlinks
  # Now includes newlib-nano and libgloss-htif support
  riscv-embedded-gcc = pkgs.symlinkJoin {
    name = "riscv64-unknown-elf-gcc";
    paths = [ pkgsCrossWithNano.buildPackages.gcc libgloss-htif ];
    postBuild = ''
      cd $out/bin
      for f in riscv64-none-elf-*; do
        [ -e "$f" ] || continue
        newname=''${f/riscv64-none-elf/riscv64-unknown-elf}
        ln -sf "$f" "$newname"
      done
    '';
  };

  # RISC-V Linux toolchain
  riscv-linux-gcc = let
    cc = pkgs.pkgsCross.riscv64.stdenv.cc;
    libcStatic = pkgs.pkgsCross.riscv64.stdenv.cc.libc.static;
  in pkgs.runCommand "riscv64-linux-gnu-toolchain" {} ''
    mkdir -p $out/bin
    for f in ${cc}/bin/riscv64-unknown-linux-gnu-*; do
      [ -e "$f" ] || continue
      name=$(basename "$f")
      echo '#!${pkgs.stdenv.shell}' > $out/bin/$name
      echo 'exec "'"$f"'" -L${libcStatic}/lib "$@"' >> $out/bin/$name
      chmod +x $out/bin/$name
    done
  '';


}
