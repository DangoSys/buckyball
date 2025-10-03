set(BUCKYBALL_DEEPSEEKR1_DIR ${BUCKYBALL_DIR}/DeepSeekR1)
set(BUCKYBALL_LENET_DIR ${BUCKYBALL_DIR}/LeNet)
set(BUCKYBALL_RESNET_DIR ${BUCKYBALL_DIR}/ResNet)
set(BUCKYBALL_MOBILENETV3_DIR ${BUCKYBALL_DIR}/MobileNetV3)
set(BUCKYBALL_BERT_DIR ${BUCKYBALL_DIR}/Bert)
set(BUCKYBALL_STABLEDIFFUSION_DIR ${BUCKYBALL_DIR}/StableDiffusion)

add_subdirectory(LeNet)
add_subdirectory(ResNet18)
add_subdirectory(MobileNetV3)
add_subdirectory(Bert)
add_subdirectory(StableDiffusion)

add_custom_target(buckyball-model-build DEPENDS
  BuckyballLENET
  BuckyballRESNET18
  BuckyballMOBILENETV3
  BuckyballBERT
  BuckyballSTABLEDIFFUSION
)
