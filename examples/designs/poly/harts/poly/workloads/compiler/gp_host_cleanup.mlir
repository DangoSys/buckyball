// CHECK-NOT: @gp_space
// CHECK-NOT: @operator
// CHECK-LABEL: func.func @prepare_37()
// CHECK-LABEL: func.func @head_attention_out_9()
memref.global @gp_space : memref<16xi8> {bbacc.gp_space}
func.func private @operator_0() attributes {bbacc.gp_operator} {
  %a = memref.get_global @gp_space : memref<16xi8>
  return
}
func.func @operator() attributes {bbacc.gp_entry} {
  func.call @operator_0() : () -> ()
  return
}
func.func @prepare_37() { return }
func.func @head_attention_out_9() { return }
