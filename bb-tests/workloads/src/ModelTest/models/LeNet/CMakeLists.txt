set(LENET_MODEL_PATH ${MODEL_LENET_DIR})
set(IMG_PATH ${MODEL_LENET_DIR}/images)


add_custom_command(
  OUTPUT ${LENET_MODEL_PATH}/lenet-model.pth
  COMMAND export LENET_DIR=${MODEL_LENET_DIR} &&
          cd ${MODEL_LENET_DIR} &&
          python3 ${MODEL_LENET_DIR}/pytorch-lenet-train.py
  DEPENDS pytorch-lenet-train.py model.py
  COMMENT "Generating lenet-model.pth"
  USES_TERMINAL
)

add_custom_command(
  OUTPUT  ${MODEL_LENET_DIR}/forward.mlir ${MODEL_LENET_DIR}/subgraph0.mlir ${MODEL_LENET_DIR}/arg0.data
  COMMAND export LENET_DIR=${MODEL_LENET_DIR} &&
          python3 ${MODEL_LENET_DIR}/buddy-lenet-import.py
  DEPENDS buddy-lenet-import.py model.py ${LENET_MODEL_PATH}/lenet-model.pth
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_target(lenet-model-build DEPENDS
  ${MODEL_LENET_DIR}/forward.mlir
  ${MODEL_LENET_DIR}/subgraph0.mlir
  ${MODEL_LENET_DIR}/arg0.data
)
