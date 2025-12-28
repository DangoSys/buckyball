package framework.gpdomain.sequencer.decoder

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.core.rocket.RoCCCommandBB
import framework.gpdomain.sequencer.decoder.{Decoder, DecoderParam}
import chisel3.experimental.hierarchy.{instantiable, public}

/**
 * Domain Decoder Parameter Object
 * Contains the configuration for RVV instruction decoding
 */
object DomainDecoderParameter {

  // Get all RVV instructions from our local database
  lazy val allInstructions: Seq[InstructionEncoding.Instruction] = {
    RVVInstructions.allInstructions
  }

  // Create decoder parameter using our local instruction types
  lazy val decoderParam: DecoderParam = DecoderParam(
    fpuEnable = true,  // Enable floating-point vector instructions
    zvbbEnable = true, // Enable vector bit manipulation
    useXsfmm = false,  // Disable xsfmm extension for now
    allInstructions = allInstructions
  )

}

/**
 * Domain Decoder IO
 */
class DomainDecoderIO(b: GlobalConfig) extends Bundle {
  val inst_i    = Input(new RoCCCommandBB(b.core.xLen))
  val decoded_o = Decoder.bundle(DomainDecoderParameter.decoderParam).cloneType
}

/**
 * Domain Decoder Module
 * Encapsulates the T1 decoder logic with local instruction database
 */
@instantiable
class DomainDecoder(val b: GlobalConfig) extends Module {
  @public
  val io = IO(new DomainDecoderIO(b))

  import DomainDecoderParameter._

  // Instantiate the T1 decoder with our local instructions
  val decode = Decoder.decode(decoderParam)

  // Decode the incoming instruction
  // RoCCCommandBB has 'inst' field (RoCCInstruction Bundle), need to convert to UInt
  val inst = io.inst_i.inst.asUInt
  io.decoded_o := decode(inst)

}
