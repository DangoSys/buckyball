set(GEMMINI_LENET_DIR ${GEMMINI_DIR}/LeNet)
set(GEMMINI_RESNET18_DIR ${GEMMINI_DIR}/ResNet18)
set(GEMMINI_MOBILENETV3_DIR ${GEMMINI_DIR}/MobileNetV3)
set(GEMMINI_BERT_DIR ${GEMMINI_DIR}/Bert)
set(GEMMINI_STABLEDIFFUSION_DIR ${GEMMINI_DIR}/StableDiffusion)
set(GEMMINI_WHISPER_DIR ${GEMMINI_DIR}/Whisper)
set(GEMMINI_LLAMA2_DIR ${GEMMINI_DIR}/Llama2)
set(GEMMINI_DEEPSEEKR1_DIR ${GEMMINI_DIR}/DeepSeekR1)

add_subdirectory(LeNet)
add_subdirectory(ResNet18)
add_subdirectory(MobileNetV3)
add_subdirectory(Bert)
add_subdirectory(StableDiffusion)
add_subdirectory(llama2)
add_subdirectory(DeepSeekR1)

# add_custom_target(gemmini-model-build DEPENDS
#   GemminiLENET
#   GemminiRESNET18
#   GemminiMOBILENETV3
#   GemminiBERT
#   GemminiSTABLEDIFFUSION
#   GemminiLLAMA2
#   GemminiDEEPSEEKR1
# )
