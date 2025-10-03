add_custom_command(
  OUTPUT ${MODEL_BERT_DIR}/forward.mlir
         ${MODEL_BERT_DIR}/subgraph0.mlir
         ${MODEL_BERT_DIR}/arg0.data
         ${MODEL_BERT_DIR}/arg1.data
  COMMAND ${Python3_EXECUTABLE} ${MODEL_BERT_DIR}/import-bert.py
          --output-dir ${MODEL_BERT_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_target(bert-model-build DEPENDS
  ${MODEL_BERT_DIR}/forward.mlir
  ${MODEL_BERT_DIR}/subgraph0.mlir
  ${MODEL_BERT_DIR}/arg0.data
  ${MODEL_BERT_DIR}/arg1.data
)
