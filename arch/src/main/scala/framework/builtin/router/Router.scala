package framework.builtin.router

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

/**
 * Router:
 *
 * Input: valid signals for each input channel
 * Output: source input channel ID for each output channel
 */
@instantiable
class Router(val numInputs: Int, val numOutputs: Int) extends Module {
  val inputChannelIdWidth  = log2Ceil(numInputs)
  val outputChannelIdWidth = log2Ceil(numOutputs)

  @public
  val io = IO(new Bundle {
    val in  = Input(Vec(numInputs, Bool()))
    val out = Vec(numOutputs, Decoupled(UInt(inputChannelIdWidth.W)))
  })

  // For each output channel, if ready, select an input with priority
  for (outIdx <- 0 until numOutputs) {
    // Count how many inputs are valid
    val numValidInputs = PopCount(io.in)

    // Select one input with priority from all valid inputs
    val selectedIdx   = PriorityEncoder(io.in)
    val hasValidInput = numValidInputs > 0.U

    io.out(outIdx).valid := hasValidInput && io.out(outIdx).ready
    io.out(outIdx).bits  := selectedIdx.asUInt.asTypeOf(io.out(outIdx).bits.cloneType)
  }
}
