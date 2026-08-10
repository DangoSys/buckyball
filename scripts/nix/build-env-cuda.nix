{ pkgs }:

let
  # Host driver reports CUDA 12.8 (570.x); do not use default 12.9+.
  cudaPkgs = pkgs.cudaPackages_12_8;
in
{
  cudatoolkit = cudaPkgs.cudatoolkit;
  nvcc = cudaPkgs.cuda_nvcc;
  cudart = cudaPkgs.cuda_cudart;

  # CUDA host compiler: default nix gcc is 15, toolkit wants <=13.
  # Versioned names only — avoids colliding with default gcc on PATH.
  gcc13 = pkgs.runCommand "gcc13-named" { } ''
    mkdir -p $out/bin
    for b in gcc g++ c++ cpp; do
      [ -e ${pkgs.gcc13}/bin/$b ] || continue
      ln -s ${pkgs.gcc13}/bin/$b $out/bin/$b-13
    done
  '';
}
