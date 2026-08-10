package framework.memdomain.isa

import framework.system.core.rocket.BuckyballCommand

/** Shared MemDomain ISA encoding helpers. */
object MemISA {
  val MsetFunct = 0x20

  final case class MsetArgs(
    bankId:  Int,
    row:     Int = 0,
    columns: Int,
    alloc:   Boolean,
    clear:   Boolean)

  def mset(args: MsetArgs): BuckyballCommand = {
    require(args.bankId >= 0 && args.bankId < (1 << 10), s"MSET bank ID out of range: ${args.bankId}")
    require(args.row >= 0 && args.row < (1 << 5), s"MSET row out of range: ${args.row}")
    require(args.columns >= 0 && args.columns <= 32, s"MSET column count out of range: ${args.columns}")
    require(!args.clear || args.alloc, "MSET clear requires allocation")

    val rs1 = BigInt(args.bankId)
    val rs2 =
      BigInt(args.row) |
        (BigInt(args.columns) << 5) |
        (if (args.alloc) BigInt(1) << 10 else 0) |
        (if (args.clear) BigInt(1) << 11 else 0)
    BuckyballCommand(MsetFunct, rs1, rs2)
  }

}
