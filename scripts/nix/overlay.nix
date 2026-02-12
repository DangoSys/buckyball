{ bebopPkgs }:

final: prev:
{
  riscv = final.callPackage ./build-env-riscv.nix { };
  tools = final.callPackage ./build-env-tools.nix { };
  bebop = final.callPackage ./build-env-bebop.nix { inherit bebopPkgs; };
  bbdev = final.callPackage ./build-env-bbdev.nix { };
  python = final.callPackage ./build-env-python.nix { };
  scala = final.callPackage ./build-env-scala.nix { };
  doc = final.callPackage ./build-env-doc.nix { };
}
