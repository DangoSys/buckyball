add_custom_command(
  OUTPUT ${MODEL_STABLEDIFFUSION_DIR}/arg0_text_encoder.data
         ${MODEL_STABLEDIFFUSION_DIR}/arg1_text_encoder.data
         ${MODEL_STABLEDIFFUSION_DIR}/arg0_unet.data
         ${MODEL_STABLEDIFFUSION_DIR}/arg0_vae.data
         ${MODEL_STABLEDIFFUSION_DIR}/forward_text_encoder.mlir
         ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_text_encoder.mlir
         ${MODEL_STABLEDIFFUSION_DIR}/forward_unet.mlir
         ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
         ${MODEL_STABLEDIFFUSION_DIR}/forward_vae.mlir
         ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
  COMMAND ${Python3_EXECUTABLE} ${MODEL_STABLEDIFFUSION_DIR}/import-stable-diffusion.py
  --output-dir ${MODEL_STABLEDIFFUSION_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_target(stablediffusion-model-build DEPENDS
  ${MODEL_STABLEDIFFUSION_DIR}/arg0_text_encoder.data
  ${MODEL_STABLEDIFFUSION_DIR}/arg1_text_encoder.data
  ${MODEL_STABLEDIFFUSION_DIR}/arg0_unet.data
  ${MODEL_STABLEDIFFUSION_DIR}/arg0_vae.data
  ${MODEL_STABLEDIFFUSION_DIR}/forward_text_encoder.mlir
  ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_text_encoder.mlir
  ${MODEL_STABLEDIFFUSION_DIR}/forward_unet.mlir
  ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
  ${MODEL_STABLEDIFFUSION_DIR}/forward_vae.mlir
  ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
)
