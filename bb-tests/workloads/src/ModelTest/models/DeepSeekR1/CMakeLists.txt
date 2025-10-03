add_custom_command(
  OUTPUT ${MODEL_DEEPSEEKR1_DIR}/forward.mlir
         ${MODEL_DEEPSEEKR1_DIR}/subgraph0.mlir
         ${MODEL_DEEPSEEKR1_DIR}/arg0.data
  COMMAND python3 ${MODEL_DEEPSEEKR1_DIR}/import-deepseek-r1.py
          --output-dir ${MODEL_DEEPSEEKR1_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)

add_custom_target(deepseekr1-model-build DEPENDS
  ${MODEL_DEEPSEEKR1_DIR}/forward.mlir
  ${MODEL_DEEPSEEKR1_DIR}/subgraph0.mlir
  ${MODEL_DEEPSEEKR1_DIR}/arg0.data
)
