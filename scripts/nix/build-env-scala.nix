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
      cp $src $out/bin/mill
      chmod +x $out/bin/mill
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

  # sbt 1.8.2
  sbt = (pkgs.sbt.override { jre = pkgs.jdk17; }).overrideAttrs (old: {
    version = "1.8.2";
    src = pkgs.fetchurl {
      url = "https://github.com/sbt/sbt/releases/download/v1.8.2/sbt-1.8.2.tgz";
      sha256 = "11j6vyxpiqbaxg5pzm6awmrdf6fkz3pw14zszrnxdnvll16k8r8z";
    };
  });

  # Scala formatter
  scalafmt = pkgs.scalafmt;

  # Coursier - Scala dependency manager and launcher
  coursier = pkgs.coursier;
}
