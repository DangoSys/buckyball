#pragma once
#include <stddef.h>
#include <stdint.h>

namespace rvv_flash {
constexpr unsigned queries = 16, keys = 64, lanes = 16;
constexpr unsigned maskOffset = 128;
constexpr unsigned stateOffset = maskOffset + queries * lanes * sizeof(float);
struct State {
  float sum[queries], maximum[queries];
  float scores[queries * keys];
  float blockMaximum[queries], blockSum[queries];
  float alpha[queries], beta[queries], mergedMaximum[queries];
};
} // namespace rvv_flash
