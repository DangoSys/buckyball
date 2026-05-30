set(E2E_MODEL_DIR ${MODELTEST_WORKLOAD_DIR}/e2e)

# Enable all architectures and models by default for e2e
# Users can override with -DARCH=... -DMODEL=... at cmake configure time
if(NOT DEFINED ARCH)
  set(ARCH_GEMMINI ON)
  set(ARCH_BUCKYBALL ON)
endif()

if(NOT DEFINED MODEL)
  set(MODEL_LENET ON)
  set(MODEL_RESNET18 ON)
  set(MODEL_MOBILENETV3 ON)
  set(MODEL_BERT ON)
  set(MODEL_WHISPER ON)
  set(MODEL_STABLEDIFFUSION ON)
  set(MODEL_LLAMA2 ON)
  set(MODEL_DEEPSEEKR1 ON)
  set(MODEL_QWEN3 ON)
  set(MODEL_YOLO26 ON)
  set(MODEL_BUDDYNEXT ON)
  set(MODEL_GEMMA4 ON)
endif()

# EXCLUDE_FROM_ALL keeps e2e targets out of the default `ninja` build but
# they remain buildable explicitly: `ninja buddy-gemmini-lenet-run`
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/e2e/models/CMakeLists.txt")
  add_subdirectory(e2e/models EXCLUDE_FROM_ALL)
endif()

unset(ARCH CACHE)
unset(MODEL CACHE)
