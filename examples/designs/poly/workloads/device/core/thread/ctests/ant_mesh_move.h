#pragma once
#include <stdint.h>

enum ant_mesh_move_role { MESH_FILL, MESH_MOVE, MESH_CONSUME };

struct ant_mesh_move_args {
  uint64_t input;
  uint64_t output;
  uint64_t sync;
  uint64_t role;
};
