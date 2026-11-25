{ pkgs }:

{
  rsync = pkgs.rsync;
  # Official nodejs.org build: node-addon-require-builtin (DeepSeek Harness)
  # pattern-matches machine code of the node binary and fails on nixpkgs builds.
  nodejs = pkgs.stdenv.mkDerivation rec {
    pname = "nodejs-bin";
    version = "24.18.0";
    src = pkgs.fetchurl {
      url = "https://nodejs.org/dist/v${version}/node-v${version}-linux-x64.tar.xz";
      hash = "sha256-VapxU/nYjyjXZfza1a5pRbXA+Yo2iBcDgX5MRQ+nZ0I=";
    };
    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib ];
    dontStrip = true;
    installPhase = ''
      mkdir -p $out
      cp -r bin include lib share $out/
    '';
  };
  git = pkgs.git;
  pnpm = pkgs.pnpm;
  clang = pkgs.clang;
}
