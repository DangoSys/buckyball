{ pkgs }:

let
  # bazelisk as both `bazel` and `bazelisk`; version comes from repo .bazelversion.
  bazel = pkgs.runCommand "bazel" { } ''
    mkdir -p $out/bin
    ln -s ${pkgs.bazelisk}/bin/bazelisk $out/bin/bazelisk
    ln -s ${pkgs.bazelisk}/bin/bazelisk $out/bin/bazel
  '';
in
{
  rsync = pkgs.rsync;
  nodejs = pkgs.nodejs;
  git = pkgs.git;
  pnpm = pkgs.pnpm;
  clang = pkgs.clang;
  inherit bazel;
}
