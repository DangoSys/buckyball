set(GEMMINI_DEEPSEEKR1_DIR ${GEMMINI_DIR}/DeepSeekR1)
set(GEMMINI_LENET_DIR ${GEMMINI_DIR}/LeNet)
set(GEMMINI_RESNET18_DIR ${GEMMINI_DIR}/ResNet18)

add_subdirectory(LeNet)
add_subdirectory(ResNet18)

add_custom_target(gemmini-model-build DEPENDS
  GemminiLENET
  GemminiRESNET18
)
