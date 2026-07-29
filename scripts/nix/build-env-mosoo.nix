{ pkgs }:

let
  workerdVersion = "1.20260714.1";
  runtimeLibraryPath = pkgs.lib.makeLibraryPath (with pkgs; [
    glibc
    stdenv.cc.cc
  ]);
  dynamicLinker = pkgs.stdenv.cc.bintools.dynamicLinker;
  workerd = pkgs.stdenv.mkDerivation {
    pname = "workerd";
    version = workerdVersion;

    src = pkgs.fetchurl {
      url = "https://registry.npmjs.org/@cloudflare/workerd-linux-64/-/workerd-linux-64-${workerdVersion}.tgz";
      sha256 = "0z4kijx9bri6jfbfci09ivip7wh4s6x5j9rhlj78iifjnzjdfz9z";
    };

    nativeBuildInputs = [ pkgs.patchelf ];
    dontBuild = true;

    installPhase = ''
      runHook preInstall

      mkdir -p $out/bin
      cp bin/workerd $out/bin/workerd
      chmod +x $out/bin/workerd
      patchelf \
        --set-interpreter ${dynamicLinker} \
        --set-rpath ${runtimeLibraryPath} \
        $out/bin/workerd

      runHook postInstall
    '';
  };
in
{
  # Mosoo local development entry points. Project package tools such as
  # wrangler/vite-plus are resolved from mosoo's bun workspace dependencies.
  bun = pkgs.bun;
  just = pkgs.just;
  workerd = workerd;
}
