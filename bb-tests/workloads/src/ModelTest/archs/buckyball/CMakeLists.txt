set(BUCKYBALL_DEEPSEEKR1_DIR ${BUCKYBALL_DIR}/DeepSeekR1)
set(BUCKYBALL_LENET_DIR ${BUCKYBALL_DIR}/LeNet)
set(BUCKYBALL_RESNET_DIR ${BUCKYBALL_DIR}/ResNet)

add_subdirectory(LeNet)

add_custom_target(buckyball-model-build DEPENDS
  BuckyballLENET
)
