#include "attention-parameters.h"
#include <algorithm>
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

using Hidden = MemRef<float, 3>;
using Cache = MemRef<float, 4>;
using Slots = MemRef<int64_t, 1>;
using Positions = MemRef<int64_t, 2>;
using Floats = MemRef<float, 1>;
using Bytes = MemRef<int8_t, 1>;
struct Result {
  Hidden hidden;
  Cache keys, values;
};
extern "C" void _mlir_ciface_forward_prefill(Result *, Floats *, Bytes *,
                                             Hidden *, Cache *, Cache *,
                                             Slots *, Positions *);
extern "C" void _mlir_ciface_forward_decode(Result *, Floats *, Bytes *,
                                            Hidden *, Cache *, Cache *, Slots *,
                                            Positions *);

template <typename T, size_t Rank>
void load(const std::string &path, MemRef<T, Rank> &tensor) {
  std::ifstream stream;
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  stream.open(path, std::ios::binary);
  stream.read(reinterpret_cast<char *>(tensor.getData()),
              tensor.getSize() * sizeof(T));
}

template <size_t Rank>
void compare(const std::string &path, MemRef<float, Rank> &actual) {
  auto name = std::filesystem::path(path);
  std::ofstream saved("actual-" + name.parent_path().filename().string() + "-" +
                          name.filename().string(),
                      std::ios::binary);
  saved.exceptions(std::ios::failbit | std::ios::badbit);
  saved.write(reinterpret_cast<char *>(actual.getData()),
              actual.getSize() * sizeof(float));
  saved.close();
  std::vector<size_t> shape(actual.getSizes(), actual.getSizes() + Rank);
  MemRef<float, Rank> expected(shape);
  load(path, expected);
  for (size_t i = 0; i < actual.getSize(); ++i)
    if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) >
                                         2e-5f + 2e-4f * std::abs(expected[i]))
      throw std::runtime_error("attention mismatch: " + path +
                               " index=" + std::to_string(i) +
                               " actual=" + std::to_string(actual[i]) +
                               " expected=" + std::to_string(expected[i]));
}

int main(int argc, char **argv) {
  const auto fixtures =
      std::filesystem::path(argv[0]).parent_path() / "attention";
  runtime_init(1024 * 1024);
  void *workspace = aligned_alloc(64, 64 * 1024 * 1024);
  if (!workspace)
    throw std::bad_alloc();
  workspace_init(workspace, 64 * 1024 * 1024);
  const std::vector<size_t> cacheShape{1, kvHeads, capacity, headSize};
  Cache keys(cacheShape, 0.0f), values(cacheShape, 0.0f);
  for (bool prefill : {true, false}) {
    workspace_begin(workspace, 64 * 1024 * 1024);
    size_t count = prefill ? 3 : 1;
    std::string path = (fixtures / (prefill ? "prefill" : "decode")).string();
    Hidden hidden({1, count, hiddenSize});
    Slots slots({count});
    Positions positions({3, count});
    Floats floats({prefill ? prefillFloats : decodeFloats});
    Bytes bytes({prefill ? prefillBytes : decodeBytes});
    load(path + "/hidden.bin", hidden);
    load(path + "/cache_positions.bin", slots);
    load(path + "/positions.bin", positions);
    load(path + "/params.f32", floats);
    load(path + "/weights.bin", bytes);
    if (!prefill) {
      compare((fixtures / "prefill/expected-keys.bin").string(), keys);
      compare((fixtures / "prefill/expected-values.bin").string(), values);
    }
    Result result{Hidden({1, count, hiddenSize}, false, 0),
                  Cache(cacheShape, false, 0), Cache(cacheShape, false, 0)};
    auto run =
        prefill ? _mlir_ciface_forward_prefill : _mlir_ciface_forward_decode;
    run(&result, &floats, &bytes, &hidden, &keys, &values, &slots, &positions);
    compare(path + "/expected-keys.bin", result.keys);
    compare(path + "/expected-values.bin", result.values);
    compare(path + "/expected-hidden.bin", result.hidden);
    std::copy_n(result.keys.getData(), keys.getSize(), keys.getData());
    std::copy_n(result.values.getData(), values.getSize(), values.getData());
    workspace_free(result.hidden.release());
    workspace_free(result.keys.release());
    workspace_free(result.values.release());
    std::cout << "OMNI MXFP8 ATTENTION " << (prefill ? "PREFILL" : "DECODE")
              << " PASSED\n";
  }
  free(workspace);
}
