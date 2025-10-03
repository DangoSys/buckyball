add_custom_command(
  OUTPUT ${MODEL_RESNET18_DIR}/arg0.data
         ${MODEL_RESNET18_DIR}/forward.mlir
         ${MODEL_RESNET18_DIR}/subgraph0.mlir
  COMMAND python3 ${MODEL_RESNET18_DIR}/buddy-resnet-import.py
          --output-dir ${MODEL_RESNET18_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_target(resnet-model-build DEPENDS
  ${MODEL_RESNET18_DIR}/forward.mlir
  ${MODEL_RESNET18_DIR}/subgraph0.mlir
  ${MODEL_RESNET18_DIR}/arg0.data
)
