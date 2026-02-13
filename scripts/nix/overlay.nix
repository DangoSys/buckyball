final: prev:
{
  bbdev = final.callPackage ./build-env-bbdev.nix { };
  bebop = final.callPackage ./build-env-bebop.nix { };
  clibs = final.callPackage ./build-env-clibs.nix { };
  doc = final.callPackage ./build-env-doc.nix { };
  python = final.callPackage ./build-env-python.nix { };
  riscv = final.callPackage ./build-env-riscv.nix { };
  scala = final.callPackage ./build-env-scala.nix { };
  tools = final.callPackage ./build-env-tools.nix { };
}
