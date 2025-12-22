// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2024 Mio

package framework.gpdomain.sequencer.decoder

/**
 * Local Instruction Database
 * Provides instruction definitions without external dependencies
 */
object InstructionEncoding {

  /** Like chisel3.BitPat, this stores the instruction encoding */
  case class Encoding(value: BigInt, mask: BigInt) {
    def toBitMask(bit_pat: String) =
      Seq.tabulate(32)(i => if (!mask.testBit(i)) bit_pat else if (value.testBit(i)) "1" else "0").reverse.mkString
    override def toString: String = toBitMask("?")
  }

  object Encoding {
    implicit val rw: upickle.default.ReadWriter[Encoding] = upickle.default.macroRW

    def fromString(str: String): Encoding = {
      require(str.length == 32, s"Encoding string must be 32 bits, got ${str.length}")
      Encoding(
        str.reverse.zipWithIndex.map {
          case (c, i) =>
            c match {
              case '1' => BigInt(1) << i
              case '0' => BigInt(0)
              case '?' => BigInt(0)
              case _   => throw new IllegalArgumentException(s"Invalid encoding character: $c")
            }
        }.sum,
        str.reverse.zipWithIndex.map {
          case (c, i) =>
            c match {
              case '1' => BigInt(1) << i
              case '0' => BigInt(1) << i
              case '?' => BigInt(0)
              case _   => throw new IllegalArgumentException(s"Invalid encoding character: $c")
            }
        }.sum
      )
    }

  }

  case class Arg(name: String, msb: Int, lsb: Int) {
    override def toString: String = name
  }

  object Arg {
    implicit val rw: upickle.default.ReadWriter[Arg] = upickle.default.macroRW
  }

  case class InstructionSet(name: String)

  object InstructionSet {
    implicit val rw: upickle.default.ReadWriter[InstructionSet] = upickle.default.macroRW
  }

  case class Instruction(
    name:            String,
    encoding:        Encoding,
    args:            Seq[Arg],
    instructionSets: Seq[InstructionSet],
    pseudoFrom:      Option[Instruction],
    ratified:        Boolean,
    custom:          Boolean) {
    def instructionSet: InstructionSet = instructionSets.head
  }

  object Instruction {
    implicit val rw: upickle.default.ReadWriter[Instruction] = upickle.default.macroRW
  }

}

/**
 * Manual RVV Instruction Database
 * Contains commonly used RVV instructions with their encodings
 */
object RVVInstructions {
  import InstructionEncoding._

  // Common argument definitions
  val vd     = Arg("vd", 11, 7)
  val vs1    = Arg("vs1", 19, 15)
  val vs2    = Arg("vs2", 24, 20)
  val vs3    = Arg("vs3", 11, 7)
  val vm     = Arg("vm", 25, 25)
  val rs1    = Arg("rs1", 19, 15)
  val rs2    = Arg("rs2", 24, 20)
  val rd     = Arg("rd", 11, 7)
  val zimm5  = Arg("zimm5", 19, 15)
  val zimm10 = Arg("zimm10", 29, 20)
  val zimm11 = Arg("zimm11", 30, 20)
  val simm5  = Arg("simm5", 19, 15)
  val nf     = Arg("nf", 31, 29)

  // Instruction set definitions
  val rv_v    = InstructionSet("rv_v")
  val rv_zvbb = InstructionSet("rv_zvbb")

  /**
   * Create all RVV instructions
   * Returns a sequence of Instructions for the decoder
   */
  def allInstructions: Seq[Instruction] = {
    configInstructions ++
      integerArithmeticInstructions ++
      loadStoreInstructions ++
      maskInstructions ++
      permuteInstructions
  }

  // ============================================================================
  // Vector Configuration Instructions
  // ============================================================================
  def configInstructions: Seq[Instruction] = Seq(
    Instruction(
      name = "vsetvli",
      encoding =
        Encoding.fromString(
          "0????????????????111?????1010111"
        ), // 31=0, zimm11(30..20), rs1(19..15), funct3=111(14..12), rd(11..7), opcode=1010111(6..0)
      args = Seq(rd, rs1, zimm11),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vsetivli",
      encoding =
        Encoding.fromString(
          "11???????????????111?????1010111"
        ), // 31..30=11, zimm10(29..20), zimm5(19..15), funct3=111(14..12), rd(11..7), opcode=1010111(6..0)
      args = Seq(rd, zimm10, zimm5),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vsetvl",
      encoding =
        Encoding.fromString(
          "1000000??????????111?????1010111"
        ), // 31=1, 30..25=000000, rs2(24..20), rs1(19..15), funct3=111(14..12), rd(11..7), opcode=1010111(6..0)
      args = Seq(rd, rs1, rs2),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    )
  )

  // ============================================================================
  // Integer Arithmetic Instructions (Common ones)
  // ============================================================================
  def integerArithmeticInstructions: Seq[Instruction] = Seq(
    // VADD variants
    Instruction(
      name = "vadd.vv",
      encoding = Encoding.fromString("000000???????????000?????1010111"), // funct6=000000, funct3=000 (OPIVV)
      args = Seq(vd, vs2, vs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vadd.vx",
      encoding = Encoding.fromString("000000???????????100?????1010111"), // funct6=000000, funct3=100 (OPIVX)
      args = Seq(vd, vs2, rs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vadd.vi",
      encoding = Encoding.fromString("000000???????????011?????1010111"), // funct6=000000, funct3=011 (OPIVI)
      args = Seq(vd, vs2, simm5, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    // VSUB variants
    Instruction(
      name = "vsub.vv",
      encoding = Encoding.fromString("000010???????????000?????1010111"), // funct6=000010, funct3=000
      args = Seq(vd, vs2, vs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vsub.vx",
      encoding = Encoding.fromString("000010???????????100?????1010111"), // funct6=000010, funct3=100
      args = Seq(vd, vs2, rs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    // VMUL variants
    Instruction(
      name = "vmul.vv",
      encoding = Encoding.fromString("100101???????????010?????1010111"), // funct6=100101, funct3=010 (OPMVV)
      args = Seq(vd, vs2, vs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vmul.vx",
      encoding = Encoding.fromString("100101???????????110?????1010111"), // funct6=100101, funct3=110 (OPMVX)
      args = Seq(vd, vs2, rs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    )
  )

  // ============================================================================
  // Load/Store Instructions
  // ============================================================================
  def loadStoreInstructions: Seq[Instruction] = Seq(
    // Unit-stride loads
    Instruction(
      name = "vle8.v",
      encoding =
        Encoding.fromString(
          "???000?00000?????000?????0000111"
        ),                                                                // nf, mew=0, mop=00, lumop=00000, funct3=000, opcode=0000111
      args = Seq(vd, rs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vle16.v",
      encoding = Encoding.fromString("???000?00000?????101?????0000111"), // funct3=101
      args = Seq(vd, rs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vle32.v",
      encoding = Encoding.fromString("???000?00000?????110?????0000111"), // funct3=110
      args = Seq(vd, rs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vle64.v",
      encoding = Encoding.fromString("???000?00000?????111?????0000111"), // funct3=111
      args = Seq(vd, rs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    // Unit-stride stores
    Instruction(
      name = "vse8.v",
      encoding = Encoding.fromString("???000?00000?????000?????0100111"), // opcode=0100111 (VS)
      args = Seq(vs3, rs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vse16.v",
      encoding = Encoding.fromString("???000?00000?????101?????0100111"),
      args = Seq(vs3, rs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vse32.v",
      encoding = Encoding.fromString("???000?00000?????110?????0100111"),
      args = Seq(vs3, rs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vse64.v",
      encoding = Encoding.fromString("???000?00000?????111?????0100111"),
      args = Seq(vs3, rs1, vm),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    )
  )

  // ============================================================================
  // Mask Instructions
  // ============================================================================
  def maskInstructions: Seq[Instruction] = Seq(
    Instruction(
      name = "vmand.mm",
      encoding = Encoding.fromString("011001???????????010?????1010111"), // funct6=011001, funct3=010
      args = Seq(vd, vs2, vs1),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vmor.mm",
      encoding = Encoding.fromString("011010???????????010?????1010111"), // funct6=011010, funct3=010
      args = Seq(vd, vs2, vs1),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    )
  )

  // ============================================================================
  // Permute Instructions
  // ============================================================================
  def permuteInstructions: Seq[Instruction] = Seq(
    Instruction(
      name = "vmv.v.v",
      encoding = Encoding.fromString("010111?00000?????000?????1010111"), // funct6=010111, vs2=00000, funct3=000
      args = Seq(vd, vs1),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vmv.v.x",
      encoding = Encoding.fromString("010111?00000?????100?????1010111"), // funct6=010111, vs2=00000, funct3=100
      args = Seq(vd, rs1),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    ),
    Instruction(
      name = "vmv.v.i",
      encoding = Encoding.fromString("010111?00000?????011?????1010111"), // funct6=010111, vs2=00000, funct3=011
      args = Seq(vd, simm5),
      instructionSets = Seq(rv_v),
      pseudoFrom = None,
      ratified = true,
      custom = false
    )
  )

}
