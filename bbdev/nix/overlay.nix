# bbdev overlay
# Provides bbdev CLI package and Python dependencies
final: prev: {
  bbdevPythonPkgs = final.python312.withPackages (ps: with ps; [
    pydantic
    requests
  ]);
  bbdevCli = final.callPackage ./bbdev-install.nix { };
}
