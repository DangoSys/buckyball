final: prev:
{
  chipyard = final.callPackage ./build-env-chipyard.nix { };
  bebop = final.callPackage ./build-env-bebop.nix { };
  bbdev = final.callPackage ./build-env-bbdev.nix { };
  python = final.callPackage ./build-env-python.nix { };
  scala = final.callPackage ./build-env-scala.nix { };
  doc = final.callPackage ./build-env-doc.nix { };
}
