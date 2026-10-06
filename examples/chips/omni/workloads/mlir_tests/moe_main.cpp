#include "moe-parameters.h"
#include <buddy/Core/Container.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <runtime.h>
#include <stdexcept>
#include <string>

using Hidden = MemRef<float, 2>;
using Indices = MemRef<int64_t, 2>;
using Floats = MemRef<float, 1>;
using Bytes = MemRef<int8_t, 1>;
struct RouterResult {
  Hidden normalized;
  Indices indices;
  Hidden scores;
};
extern "C" void _mlir_ciface_forward_router(RouterResult *, Floats *, Hidden *);
extern "C" void _mlir_ciface_forward_expert(Hidden *, Bytes *, Hidden *);

template <typename T, size_t Rank>
void load(const std::string &path, MemRef<T, Rank> &tensor) {
  std::ifstream stream;
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  stream.open(path, std::ios::binary);
  stream.read(reinterpret_cast<char *>(tensor.getData()),
              tensor.getSize() * sizeof(T));
}

int main(int argc, char **argv) {
  const std::string path =
      (std::filesystem::path(argv[0]).parent_path() / "moe").string();
  runtime_init(1024 * 1024);
  void *workspace = aligned_alloc(64, 64 * 1024 * 1024);
  if (!workspace)
    throw std::bad_alloc();
  workspace_init(workspace, 64 * 1024 * 1024);
  workspace_begin(workspace, 64 * 1024 * 1024);
  Hidden input({1, hiddenSize}), expected({1, hiddenSize}), scores({1, topK});
  Indices indices({1, topK});
  Floats floats({routerFloats});
  Bytes bytes({expertBytes});
  load(path + "/input.f32", input);
  load(path + "/router/params.f32", floats);
  load(path + "/expert/weights.bin", bytes);
  RouterResult routing{Hidden({1, hiddenSize}, false, 0),
                       Indices({1, topK}, false, 0),
                       Hidden({1, topK}, false, 0)};
  _mlir_ciface_forward_router(&routing, &floats, &input);
  load(path + "/router/expected.f32", expected);
  load(path + "/router/indices.i64", indices);
  load(path + "/router/scores.f32", scores);
  for (size_t i = 0; i < hiddenSize; ++i)
    if (!std::isfinite(routing.normalized[i]) ||
        std::abs(routing.normalized[i] - expected[i]) > 1e-5f)
      throw std::runtime_error("router normalization mismatch");
  for (size_t i = 0; i < topK; ++i)
    if (routing.indices[i] != indices[i] || !std::isfinite(routing.scores[i]) ||
        std::abs(routing.scores[i] - scores[i]) > 1e-5f)
      throw std::runtime_error("router top-k mismatch");
  Hidden result({1, hiddenSize}, false, 0);
  _mlir_ciface_forward_expert(&result, &bytes, &input);
  load(path + "/expert/expected.f32", expected);
  float maximum = 0;
  for (size_t i = 0; i < hiddenSize; ++i) {
    float error = std::abs(result[i] - expected[i]);
    if (!std::isfinite(result[i]) ||
        error > 1e-5f + 1e-4f * std::abs(expected[i]))
      throw std::runtime_error("MXFP8 expert mismatch");
    maximum = std::max(maximum, error);
  }
  std::cout << "OMNI ROUTER AND MXFP8 EXPERT PASSED max_error=" << maximum
            << '\n';
  workspace_free(routing.normalized.release());
  workspace_free(routing.indices.release());
  workspace_free(routing.scores.release());
  workspace_free(result.release());
  free(workspace);
}
