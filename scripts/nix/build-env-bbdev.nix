{ pkgs }:

let
  iiiVersion = "0.10.0";
  iiiBinary = pkgs.stdenv.mkDerivation {
    pname = "iii";
    version = iiiVersion;

    src = pkgs.fetchurl {
      url = "https://github.com/iii-hq/iii/releases/download/iii/v${iiiVersion}/iii-x86_64-unknown-linux-gnu.tar.gz";
      sha256 = "sha256-XMSqKQKHL4XCQTw5r9BNbdcPS4c6HZkg4GW0XXVKg00=";
    };

    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib ];

    unpackPhase = ''
      tar xzf $src
    '';

    installPhase = ''
      mkdir -p $out/bin
      cp iii $out/bin/iii
      chmod +x $out/bin/iii
    '';
  };
in
{
  # iii engine binary
  iii = iiiBinary;

  # UV (for creating .venv and installing motia + deps)
  uv = pkgs.uv;
  # Allure CLI for sardine Allure reports
  allure = pkgs.allure;

  # Build tools (for compiling native modules)
  gcc = pkgs.gcc;
  gnumake = pkgs.gnumake;
  pkg-config = pkgs.pkg-config;
}
