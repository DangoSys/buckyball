final: prev:
{
  bbdev = final.callPackage ./build-env-bbdev.nix { };
  # Named rustTools to avoid shadowing nixpkgs `rust` (used by rust hooks)
  rustTools = final.callPackage ./build-env-rust.nix { };
  clibs = final.callPackage ./build-env-clibs.nix { };
  doc = final.callPackage ./build-env-doc.nix { };
  python = final.callPackage ./build-env-python.nix { };
  riscv = final.callPackage ./build-env-riscv.nix { };
  scala = final.callPackage ./build-env-scala.nix { };
  systemTools = final.callPackage ./build-env-system.nix { };
  tools = final.callPackage ./build-env-tools.nix { };
  kernel = final.callPackage ./build-env-kernel.nix { };
}
