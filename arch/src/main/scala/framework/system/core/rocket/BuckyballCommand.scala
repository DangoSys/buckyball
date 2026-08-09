package framework.system.core.rocket

/**
 * Elaboration-time representation of one Buckyball custom instruction.
 * Domain ISA builders own the funct7 and operand encodings; boot only streams
 * these already-formed commands.
 */
case class BuckyballCommand(funct: Int, rs1: BigInt = 0, rs2: BigInt = 0)

object BuckyballCommand {
  val Custom3Opcode = 0x7b
  val Custom3Funct3 = 3
}
