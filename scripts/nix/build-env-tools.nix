{ pkgs }:

{
  # Pin to Verilator 5.022 2024-02-24 (nixpkgs-unstable ships 5.044)
  verilator = pkgs.verilator.overrideAttrs (old: {
    version = "5.022";
    src = pkgs.fetchurl {
      url = "https://github.com/verilator/verilator/archive/refs/tags/v5.022.tar.gz";
      hash = "sha256-PC9TOPS2zn4vR6FCQBrN0Yy/TF2gYJJhjW0DbAr+8S0=";
    };
    sourceRoot = "verilator-5.022";
    doCheck = false;
  });
}
