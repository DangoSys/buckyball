set(MODEL_DEEPSEEKR1_DIR ${MODEL_DIR}/DeepSeekR1)
set(MODEL_LENET_DIR ${MODEL_DIR}/LeNet)
set(MODEL_RESNET18_DIR ${MODEL_DIR}/ResNet18)
set(MODEL_BERT_DIR ${MODEL_DIR}/Bert)
set(MODEL_MOBILENETV3_DIR ${MODEL_DIR}/MobileNetV3)
set(MODEL_WHISPER_DIR ${MODEL_DIR}/Whisper)
set(MODEL_STABLEDIFFUSION_DIR ${MODEL_DIR}/StableDiffusion)
set(MODEL_LLAMA2_DIR ${MODEL_DIR}/llama2)

# Conditionally add model subdirectories based on MODEL_* flags
if (MODEL_LENET)
  add_subdirectory(LeNet)
endif()

if (MODEL_DEEPSEEKR1)
  add_subdirectory(DeepSeekR1)
endif()

if (MODEL_RESNET18)
  add_subdirectory(ResNet18)
endif()

if (MODEL_BERT)
  add_subdirectory(Bert)
endif()

if (MODEL_MOBILENETV3)
  add_subdirectory(MobileNetV3)
endif()

if (MODEL_WHISPER)
  add_subdirectory(Whisper)
endif()

if (MODEL_STABLEDIFFUSION)
  add_subdirectory(StableDiffusion)
endif()

if (MODEL_LLAMA2)
  add_subdirectory(llama2)
endif()
