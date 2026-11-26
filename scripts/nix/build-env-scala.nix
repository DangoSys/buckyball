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

  # Chisel 6.7.0 uses CIRCT firtool 1.62.1.
  firtool = pkgs.stdenv.mkDerivation rec {
    pname = "firtool";
    version = "1.62.1";
    src = pkgs.fetchurl {
      url = "https://repo.maven.apache.org/maven2/org/chipsalliance/llvm-firtool/${version}/llvm-firtool-${version}-linux-x64.jar";
      hash = "sha256-Y6zctY7RLUwaoOze3G4kJaJS6U7vt135JTwRjX0iIDk=";
    };
    dontUnpack = true;
    nativeBuildInputs = [ pkgs.unzip pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.zlib pkgs.stdenv.cc.cc.lib ];
    installPhase = ''
      mkdir -p $out/bin
      unzip -p "$src" org.chipsalliance/llvm-firtool/linux-x64/bin/firtool > $out/bin/firtool
      chmod +x $out/bin/firtool
    '';
    meta.platforms = [ "x86_64-linux" ];
  };

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
