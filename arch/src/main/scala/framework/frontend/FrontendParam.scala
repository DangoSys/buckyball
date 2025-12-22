package framework.frontend

import chisel3.experimental.{SerializableModuleParameter}

case class FrontendParam(
  rob_entries:              Int,
  rs_out_of_order_response: Boolean)
    extends SerializableModuleParameter
