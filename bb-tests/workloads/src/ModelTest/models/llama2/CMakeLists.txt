add_custom_command(
  OUTPUT ${MODEL_LLAMA2_DIR}/forward.mlir
         ${MODEL_LLAMA2_DIR}/subgraph0.mlir
         ${MODEL_LLAMA2_DIR}/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${MODEL_LLAMA2_DIR}/import-llama2.py
          --output-dir ${MODEL_LLAMA2_DIR} --use-auth-token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  DEPENDS ${MODEL_LLAMA2_DIR}/llama2-to-hf.py
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

add_custom_target(llama2-model-build DEPENDS
  ${MODEL_LLAMA2_DIR}/forward.mlir
  ${MODEL_LLAMA2_DIR}/subgraph0.mlir
  ${MODEL_LLAMA2_DIR}/arg0.data
)
