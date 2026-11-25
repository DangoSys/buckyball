# Gemmini functional reference: Spike + libgemmini extension + riscv-pk.
# Versions follow the Chipyard commit that pins thirdparty/gemmini (8c3f9923):
# riscv-isa-sim 9c190a07, riscv-pk 9c61d298, and libgemmini ea8f7ed7 from thirdparty/gemmini.
let
  pkgs = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/1559d3daa3ecc813a650b79375ea61b6741b8746.tar.gz";
    sha256 = "sha256-LQy14TZp77TwbQf40gg1V3jo8FwJG0jGDkAH+zRHqg8=";
  }) { };
  pk = pkgs.pkgsCross.riscv64-embedded.riscv-pk.overrideAttrs {
    version = "1.0.0-9c61d298";
    src = pkgs.fetchFromGitHub {
      owner = "riscv-software-src";
      repo = "riscv-pk";
      rev = "9c61d29846d8521d9487a57739330f9682d5b542";
      hash = "sha256-jYva0aIom809y02WgEYEdeqMMjh5DcvoFVzF3nyHqkw=";
    };
  };
  libgemmini = pkgs.fetchFromGitHub {
    owner = "ucb-bar";
    repo = "libgemmini";
    rev = "ea8f7ed7afd68e001fb06ddccad9a023c990961d";
    hash = "sha256-/h5L6+9FpS0Fpzc4Sot/4NM6t9CkgvJC8EC/2DNro6c=";
  };
in
pkgs.spike.overrideAttrs {
  pname = "gemmini-spike";
  version = "1.1.1-dev-9c190a07";
  src = pkgs.fetchFromGitHub {
    owner = "riscv-software-src";
    repo = "riscv-isa-sim";
    rev = "9c190a07c6838f6392bafa4ad83acea462c7f759";
    hash = "sha256-XmTFBI1tkp0zCw4/SFhlxScYhJsoNHT1GmuhYB8qZho=";
  };
  postInstall = ''
    $CXX -shared -fPIC -O3 -std=c++17 -I $out/include -L $out/lib -Wl,-rpath,$out/lib \
      -o $out/lib/libgemmini.so ${libgemmini}/gemmini.cc
    mkdir -p $out/riscv64-unknown-elf/bin
    ln -s ${pk}/bin/pk $out/riscv64-unknown-elf/bin/pk
    cat > $out/bin/gemmini-spike <<EOF
    #!${pkgs.runtimeShell}
    exec $out/bin/spike --extlib=$out/lib/libgemmini.so --extension=gemmini "\$@"
    EOF
    chmod +x $out/bin/gemmini-spike
  '';
}
