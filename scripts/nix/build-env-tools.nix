{ pkgs }:

let
  # DRAMsim3 memory system simulator.
  dramsim3 = pkgs.stdenv.mkDerivation {
    pname = "dramsim3";
    version = "unstable-2026-07-23";
    src = pkgs.fetchFromGitHub {
      owner = "umd-memsys";
      repo = "DRAMsim3";
      rev = "29817593b3389f1337235d63cac515024ab8fd6e";
      hash = "sha256-uErpWJEn6C9oKR6Bv1NOAC3ij3ne3A6BPtjtX7D8ZwE=";
    };
    nativeBuildInputs = [ pkgs.cmake ];
    dontUseCmakeConfigure = true;
    postPatch = ''
      substituteInPlace src/dramsim3.h \
        --replace-fail '#include <string>' '#include <string>
#include <stdint.h>'
    '';
    buildPhase = ''
      runHook preBuild
      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
      cmake --build build --target dramsim3
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p $out/lib $out/include $out/share/dramsim3
      cp libdramsim3.so $out/lib/
      cp src/*.h ext/headers/*.h ext/headers/*.hpp $out/include/
      cp -r configs $out/share/dramsim3/
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

  dramsim3 = dramsim3;

  # Build acceleration tools
  ccache = pkgs.ccache;
  lld = pkgs.lld;
  cmake = pkgs.cmake;
  # Chipyard/FireSim requires Java 17 (Scala 2.12 compatibility)
  java = pkgs.jdk17;
  dtc = pkgs.dtc;
  spike = pkgs.spike;

  # Synthesis tools
  yosys = pkgs.yosys;

  # Static timing analysis
  opensta = opensta;

  # Coverage report (genhtml)
  lcov = pkgs.lcov;

  # Verilog/SystemVerilog formatter for pre-commit
  verible = pkgs.verible;
}
