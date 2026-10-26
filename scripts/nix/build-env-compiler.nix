{ pkgs }:

{
  # FlatBuffers - serialization library and compiler
  flatbuffers = pkgs.flatbuffers;

  # Header-only TOML library; device programs read their model description with it
  tomlplusplus = pkgs.tomlplusplus;

  # NUMA (Non-Uniform Memory Access) library
  numactl = pkgs.numactl;
}
