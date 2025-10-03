# ModelTest Common Libraries

This directory provides common libraries shared across different ModelTest architectures.

## Available Libraries

### 1. ModelTestCRunnerUtils
- **Purpose**: Provides MLIR runtime utilities like `memrefCopy`
- **Source**: `${MODEL_DIR}/ResNet18/CRunnerUtils.cpp`
- **Target**: Cross-platform (compiled for target architecture)

### 2. ModelTestDIP_riscv
- **Purpose**: Provides DIP operations (resize, rotate, etc.) compiled for RISC-V
- **Source**: `${BUDDY_MLIR_DIR}/frontend/Interfaces/lib/DIP.mlir`
- **Target**: RISC-V (riscv64 with +buddyext,+D)
- **Functions included**:
  - `_mlir_ciface_resize_4d_nchw_nearest_neighbour_interpolation`
  - `_mlir_ciface_resize_4d_nchw_bilinear_interpolation`
  - And other DIP operations (corr_2d, rotate, morphology, etc.)

## Usage

In your architecture-specific CMakeLists.txt:

```cmake
# Link the common libraries
set(YOUR_LIBS YourModelLib ModelTestCRunnerUtils ModelTestDIP_riscv)
target_link_libraries(your-executable ${YOUR_LIBS})
```

## Example

See `archs/gemmini/ResNet18/CMakeLists.txt` for a complete example.

## Notes

- These libraries are automatically built when you add `add_subdirectory(lib)` in the parent CMakeLists.txt
- The DIP library is specifically compiled for RISC-V with Gemmini extensions
- If you need DIP for other architectures, you can add additional custom commands following the same pattern
