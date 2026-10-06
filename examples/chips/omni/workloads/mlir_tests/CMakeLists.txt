set(REPO ${BUCKYBALL_REPO_ROOT})
set(BUDDY ${REPO}/stack/compiler/thirdparty/buddy-mlir)
set(COMPILER ${BUDDY}/build/omni)
set(MODEL_DIR ${REPO}/stack/models/archs/buckyball/omni/qwen3-omni-30b-a3b-instruct)
set(DATA ${CMAKE_CURRENT_BINARY_DIR}/moe)
set(ATTENTION_DATA ${CMAKE_CURRENT_BINARY_DIR}/attention)
set(PYTHON ${REPO}/result/bin/python3)
set(MODEL_CONFIG ${REPO}/stack/models/models/qwen3-omni-30b-a3b-instruct/configs/model.toml)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${MODEL_CONFIG})
set(CHECKPOINT_FILE ${CMAKE_CURRENT_BINARY_DIR}/checkpoint.txt)
add_custom_command(OUTPUT ${CHECKPOINT_FILE}
  COMMAND ${PYTHON} -c
    "import sys, tomllib; from pathlib import Path; from huggingface_hub import snapshot_download; p=tomllib.loads(Path(sys.argv[1]).read_text())['parameters']; Path(sys.argv[2]).write_text(snapshot_download(p['checkpoint'], revision=p['revision'], local_files_only=True))"
    ${MODEL_CONFIG} ${CHECKPOINT_FILE}
  DEPENDS ${MODEL_CONFIG}
  WORKING_DIRECTORY ${REPO} VERBATIM)
file(GLOB_RECURSE MODEL_SOURCES CONFIGURE_DEPENDS ${MODEL_DIR}/trace/*.py ${MODEL_DIR}/permute/*.py ${MODEL_DIR}/quant/*.py)
list(APPEND MODEL_SOURCES ${BUDDY}/frontend/Python/ops/tosa.py
     ${COMPILER}/python_packages/buddy/compiler/ops/tosa.py)
list(APPEND MODEL_SOURCES ${REPO}/examples/balls/mxmm/compiler/python/fp32.py)
add_custom_command(OUTPUT ${DATA}/router.mlir ${DATA}/router-forward.mlir
                          ${DATA}/expert_prefill.mlir ${DATA}/expert_prefill-forward.mlir
                          ${DATA}/expert.mlir ${DATA}/expert-forward.mlir ${DATA}/moe-parameters.h
  COMMAND bash -o pipefail -c
    "'${PYTHON}' -m stack.models.archs.buckyball.omni.qwen3-omni-30b-a3b-instruct.trace.export --checkpoint \"$(cat '${CHECKPOINT_FILE}')\" --compiler-build '${COMPILER}' --output '${DATA}'"
  DEPENDS ${CHECKPOINT_FILE} ${MODEL_SOURCES} ${REPO}/stack/models/models/qwen3-omni-30b-a3b-instruct/moe.py
  WORKING_DIRECTORY ${REPO} VERBATIM)
add_custom_command(OUTPUT ${ATTENTION_DATA}/prefill.mlir ${ATTENTION_DATA}/prefill-forward.mlir
                          ${ATTENTION_DATA}/decode.mlir ${ATTENTION_DATA}/decode-forward.mlir
                          ${ATTENTION_DATA}/attention-parameters.h
  COMMAND bash -o pipefail -c
    "'${PYTHON}' -m stack.models.archs.buckyball.omni.qwen3-omni-30b-a3b-instruct.trace.attention --checkpoint \"$(cat '${CHECKPOINT_FILE}')\" --compiler-build '${COMPILER}' --output '${ATTENTION_DATA}'"
  DEPENDS ${CHECKPOINT_FILE} ${MODEL_SOURCES} ${REPO}/stack/models/models/qwen3-omni-30b-a3b-instruct/attention.py
  WORKING_DIRECTORY ${REPO} VERBATIM)
set(FORWARD_PASSES
  -buffer-deallocation-simplification -convert-linalg-to-loops -eliminate-empty-tensors
  -llvm-request-c-wrappers -convert-math-to-llvm -convert-math-to-libm -lower-affine
  -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -expand-strided-metadata
  -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts
  -canonicalize -cse -lower-tile-runtime)
set(KERNEL_PASSES
  -eliminate-empty-tensors -convert-elementwise-to-linalg -convert-tensor-to-linalg
  -canonicalize -cse -one-shot-bufferize=bufferize-function-boundaries
  -buffer-deallocation-pipeline=private-function-dynamic-ownership=false -convert-linalg-to-tile=quantized-only
  -convert-tile-to-buckyball -lower-buckyball-to-bank-ssa -assign-physical-banks
  -llvm-request-c-wrappers -lower-bank-ssa-to-intrinsics -expand-strided-metadata
  -convert-linalg-to-loops -lower-buckyball -convert-vector-to-scf=target-rank=0
  -lower-affine -canonicalize -cse -convert-scf-to-cf -convert-cf-to-llvm
  -convert-vector-to-llvm -buffer-deallocation-simplification -bufferization-lower-deallocations
  -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm
  -finalize-memref-to-llvm -reconcile-unrealized-casts -canonicalize -cse -lower-tile-runtime)
foreach(STAGE router expert expert_prefill prefill decode)
  set(TARGET attention)
  if(STAGE MATCHES "^expert")
    set(TARGET ffn)
  endif()
  if(STAGE STREQUAL "router" OR STAGE MATCHES "^expert")
    set(STAGE_DATA ${DATA})
    set(GROUP MOE)
  else()
    set(STAGE_DATA ${ATTENTION_DATA})
    set(GROUP ATTENTION)
  endif()
  foreach(KIND kernel forward)
    if(KIND STREQUAL "kernel")
      set(NAME ${STAGE})
      string(JOIN " " PASSES ${KERNEL_PASSES})
      set(LOWER "'${COMPILER}/bin/buddy-opt' '${STAGE_DATA}/${NAME}.mlir' -pass-pipeline 'builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))' | '${COMPILER}/bin/buddy-opt' --target=${TARGET} ${PASSES}")
    else()
      set(NAME ${STAGE}-forward)
      string(JOIN " " PASSES ${FORWARD_PASSES})
      set(LOWER "'${COMPILER}/bin/buddy-opt' '${STAGE_DATA}/${NAME}.mlir' --target=${TARGET} ${PASSES}")
    endif()
    add_custom_command(OUTPUT ${NAME}.o
      COMMAND bash -o pipefail -c "${LOWER} | '${COMPILER}/bin/buddy-translate' --buddy-to-llvmir | '${BUDDY}/llvm/build/bin/opt' -passes='default<O2>' | '${COMPILER}/bin/buddy-llc' -filetype=obj -mtriple=riscv64 -O2 -mattr=+xbuckyball,+m,+D -float-abi=hard -o '${NAME}.o'"
      DEPENDS ${STAGE_DATA}/${NAME}.mlir ${COMPILER}/bin/buddy-opt VERBATIM)
    if(STAGE STREQUAL "expert_prefill")
      list(APPEND BATCH_OBJECTS ${NAME}.o)
    else()
      list(APPEND ${GROUP}_OBJECTS ${NAME}.o)
    endif()
  endforeach()
endforeach()
set(RUNTIME_PARAMS ${BUCKYBALL_ISA_DIR}/attention)
# Ball tensor tests may already provide the shared runtime and RVV kernels.
if(NOT TARGET runtime)
  add_subdirectory(${REPO}/stack/runtime runtime)
endif()
if(NOT TARGET rvv_kernels)
  add_subdirectory("${REPO}/bb-tests/workloads/lib/bbsw/kernels/rvv" rvv)
endif()
if(NOT TARGET mxfp8_pack)
  add_subdirectory("${REPO}/bb-tests/workloads/lib/bbsw/kernels/mxfp8" mxfp8)
endif()
buckyball_mlirtest_name(NAME ffn chip-moe)
add_executable(${NAME}-linux moe_main.cpp ${MOE_OBJECTS} ${DATA}/moe-parameters.h)
target_include_directories(${NAME}-linux PRIVATE ${DATA} ${BUDDY}/frontend/Interfaces)
target_link_libraries(${NAME}-linux PRIVATE runtime rvv_kernels mxfp8_pack rvv_quant_backend -static)
add_dependencies(chip-mlirtests-build ${NAME}-linux)
buckyball_mlirtest_name(NAME ffn chip-moe-batch)
add_executable(${NAME}-linux moe_batch_main.cpp ${BATCH_OBJECTS} expert.o expert-forward.o ${DATA}/moe-parameters.h)
target_include_directories(${NAME}-linux PRIVATE ${DATA} ${BUDDY}/frontend/Interfaces)
target_link_libraries(${NAME}-linux PRIVATE runtime rvv_kernels mxfp8_pack rvv_quant_backend -static)
add_dependencies(chip-mlirtests-build ${NAME}-linux)
buckyball_mlirtest_name(NAME attention chip-attention)
add_executable(${NAME}-linux attention_main.cpp ${ATTENTION_OBJECTS} ${ATTENTION_DATA}/attention-parameters.h)
target_include_directories(${NAME}-linux PRIVATE ${ATTENTION_DATA} ${BUDDY}/frontend/Interfaces)
target_link_libraries(${NAME}-linux PRIVATE runtime rvv_kernels mxfp8_pack rvv_quant_backend -static)
add_dependencies(chip-mlirtests-build ${NAME}-linux)
