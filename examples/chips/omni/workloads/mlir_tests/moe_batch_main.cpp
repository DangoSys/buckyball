#include "moe-parameters.h"
#include <algorithm>
#include <buddy/Core/Container.h>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <runtime.h>
#include <stdexcept>

using Matrix = MemRef<float, 2>;
using Bytes = MemRef<int8_t, 1>;
extern "C" void _mlir_ciface_forward_expert(Matrix *, Bytes *, Matrix *);
extern "C" void _mlir_ciface_forward_expert_prefill(Matrix *, Bytes *,
                                                    Matrix *);

template <typename T, size_t Rank>
void load(const std::filesystem::path &path, MemRef<T, Rank> &value) {
  std::ifstream stream;
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  stream.open(path, std::ios::binary);
  stream.read(reinterpret_cast<char *>(value.getData()),
              value.getSize() * sizeof(T));
}

int main(int argc, char **argv) {
  auto path = std::filesystem::path(argv[0]).parent_path() / "moe";
  runtime_init(1024 * 1024);
  void *workspace = aligned_alloc(64, 64 * 1024 * 1024);
  if (!workspace)
    throw std::bad_alloc();
  workspace_init(workspace, 64 * 1024 * 1024);
  workspace_begin(workspace, 64 * 1024 * 1024);
  Matrix input({batchSize, hiddenSize}), expected({batchSize, hiddenSize});
  Bytes weights({expertBytes});
  load(path / "batch-input.f32", input);
  load(path / "expert/weights.bin", weights);
  load(path / "expert_prefill/expected.f32", expected);
  Matrix batch({batchSize, hiddenSize}, false, 0);
  _mlir_ciface_forward_expert_prefill(&batch, &weights, &input);
  Matrix row({1, hiddenSize});
  for (size_t token = 0; token < batchSize; ++token) {
    std::copy_n(input.getData() + token * hiddenSize, hiddenSize,
                row.getData());
    Matrix single({1, hiddenSize}, false, 0);
    _mlir_ciface_forward_expert(&single, &weights, &row);
    for (size_t i = 0; i < hiddenSize; ++i) {
      size_t index = token * hiddenSize + i;
      if (!std::isfinite(batch[index]) || batch[index] != single[i] ||
          std::abs(batch[index] - expected[index]) >
              1e-5f + 1e-4f * std::abs(expected[index]))
        throw std::runtime_error(
            "MXFP8 batch differs from scalar or reference");
    }
    workspace_free(single.release());
  }
  workspace_free(batch.release());
  free(workspace);
  std::cout << "OMNI MXFP8 EXPERT BATCH PASSED\n";
}
