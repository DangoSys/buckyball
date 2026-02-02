{ pkgs }:

{
  # Fast and robust (System)Verilog simulator/compiler and linter
  verilator = pkgs.verilator;

  # RISC-V embedded toolchain (bare metal), with riscv64-unknown-elf-* symlinks
  riscv-embedded-gcc = pkgs.symlinkJoin {
    name = "riscv64-unknown-elf-gcc";
    paths = [ pkgs.pkgsCross.riscv64-embedded.buildPackages.gcc ];
    postBuild = ''
      cd $out/bin
      for f in riscv64-none-elf-*; do
        [ -e "$f" ] || continue
        newname=''${f/riscv64-none-elf/riscv64-unknown-elf}
        ln -sf "$f" "$newname"
      done
    '';
  };

  # RISC-V Linux toolchain (full stdenv.cc with static libc for -static linking)
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
