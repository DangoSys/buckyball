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

  # CUDD BDD library (required by OpenSTA)
  cudd = pkgs.stdenv.mkDerivation {
    pname = "cudd";
    version = "3.0.0";
    src = pkgs.fetchFromGitHub {
      owner = "The-OpenROAD-Project";
      repo = "cudd";
      rev = "3.0.0";
      hash = "sha256-ybsFPcggPsb6lfZbWbwxNTuZSOC7lLNY/iZSTvyFmdU=";
    };
    nativeBuildInputs = [ pkgs.autoreconfHook ];
    configureFlags = [ "--prefix=$(out)" "CFLAGS=-fPIC" "CXXFLAGS=-fPIC" ];
    installPhase = ''
      runHook preInstall
      make install
      runHook postInstall
    '';
  };

  # OpenSTA - gate-level static timing analysis
  opensta = pkgs.stdenv.mkDerivation {
    pname = "opensta";
    version = "unstable-2025";
    src = pkgs.fetchFromGitHub {
      owner = "The-OpenROAD-Project";
      repo = "OpenSTA";
      rev = "5e9e9db7061fddf1b0b9c47c49c920c56da140e3";
      hash = "sha256-SfxNh5PFWWTdTH0ZiiATV1F0qOBTh50+xM9roJMHtLg==";
    };
    nativeBuildInputs = with pkgs; [ cmake flex bison swig ];
    buildInputs = with pkgs; [ tcl eigen zlib ];
    cmakeFlags = [
      "-DCUDD_DIR=${cudd}"
      "-DUSE_TCL_READLINE=OFF"
    ];
    installPhase = ''
      runHook preInstall
      mkdir -p $out/bin
      find . -name sta -type f -executable -exec cp {} $out/bin/ \;
      runHook postInstall
    '';
  };
in
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

  dramsim2 = dramsim2;

  # Build acceleration tools
  ccache = pkgs.ccache;
  lld = pkgs.lld;
  cmake = pkgs.cmake;
  java = pkgs.jdk;
  dtc = pkgs.dtc;
  spike = pkgs.spike;

  # Synthesis tools
  yosys = pkgs.yosys;

  # Static timing analysis
  opensta = opensta;

  # Coverage report (genhtml)
  lcov = pkgs.lcov;
}
