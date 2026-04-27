{ pkgs }:

{
  # FlatBuffers - serialization library and compiler
  flatbuffers = pkgs.flatbuffers;

  # NUMA (Non-Uniform Memory Access) library
  numactl = pkgs.numactl;
}
