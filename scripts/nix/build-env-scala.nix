{ pkgs }:

let
  millVersion = "0.11.4";
  millBinary = pkgs.fetchurl {
    url = "https://github.com/com-lihaoyi/mill/releases/download/${millVersion}/${millVersion}";
    sha256 = "1swayysb1baqk7zhrlzvikd4plqznaa0nkx2bwc57dvwxp06whz2";
  };
  mill = pkgs.stdenv.mkDerivation {
    name = "mill-${millVersion}";
    src = millBinary;
    dontUnpack = true;
    nativeBuildInputs = [ pkgs.makeWrapper ];
    installPhase = ''
      mkdir -p $out/bin
      install -m755 $src $out/bin/.mill-wrapped
      makeWrapper $out/bin/.mill-wrapped $out/bin/mill \
        --set JAVA_HOME ${pkgs.jdk17} \
        --prefix PATH : ${pkgs.jdk17}/bin \
        --unset LD_LIBRARY_PATH
    '';
    meta = with pkgs.lib; {
      description = "Mill build tool ${millVersion}";
      homepage = "https://github.com/com-lihaoyi/mill";
      license = licenses.asl20;
      platforms = platforms.all;
    };
  };
in
{
  # Build tool for Scala, Java and more
  inherit mill;

  # Scala formatter - use coursier to get 2.7.5
  scalafmt = pkgs.writeShellApplication {
    name = "scalafmt";
    runtimeInputs = [ pkgs.coursier ];
    text = ''
      exec cs launch org.scalameta:scalafmt-cli_2.13:2.7.5 -- "$@"
    '';
  };

  # Coursier - Scala dependency manager and launcher
  coursier = pkgs.coursier;
}
