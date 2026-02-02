final: prev:
{
  chipyard = final.callPackage ./build-chipyard.nix { };
  bebop = final.callPackage ./build-bebop.nix { };
}
