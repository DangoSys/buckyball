{ pkgs }:

let
  # DRAMSim2 from firesim (rev matches chipyard pin); -fPIC for PIE linking under Nix
  dramsim2 = pkgs.stdenv.mkDerivation {
    pname = "dramsim2";
    version = "2023-05-10";
    src = pkgs.fetchFromGitHub {
      owner = "firesim";
      repo = "DRAMSim2";
      rev = "44322e2f935d7dac83b7adf8dd270b41a54c6acb";
      hash = "sha256-Vfb+MeWdUESc7gt6GhL6jBO1Uuvx8s1BdfhCikTyTh8=";
    };
    buildPhase = ''
      make CXXFLAGS="-DNO_STORAGE -Wall -DDEBUG_BUILD -O3 -fPIC" libdramsim.a
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p $out/lib $out/include
      cp libdramsim.a $out/lib/
      cp *.h $out/include/
      runHook postInstall
    '';
  };
in
{
  # Pin to Verilator 5.036 2025-01-01 (nixpkgs-unstable ships 5.044)
  verilator = pkgs.verilator.overrideAttrs (old: {
    version = "5.036";
    src = pkgs.fetchurl {
      url = "https://github.com/verilator/verilator/archive/refs/tags/v5.036.tar.gz";
      hash = "sha256-QZmWSILVbPahnOgMail+vjsMNeqBEGzU9yI0JZQzfEc=";
    };
    sourceRoot = "verilator-5.036";
    doCheck = false;
  });

  dramsim2 = dramsim2;

  # Build acceleration tools
  ccache = pkgs.ccache;
  lld = pkgs.lld;
}
