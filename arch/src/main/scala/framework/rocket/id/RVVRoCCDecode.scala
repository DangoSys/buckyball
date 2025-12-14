// See LICENSE.Berkeley for license details.
// See LICENSE.SiFive for license details.

package framework.rocket.id

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.rocket._
import freechips.rocketchip.rocket.Instructions._
import freechips.rocketchip.rocket.ALU._
import freechips.rocketchip.rocket.constants.ScalarOpConstants
import freechips.rocketchip.rocket.constants.MemoryOpConstants
import freechips.rocketchip.util._

/**
 * RVVRoCCDecode - Decoder for RVV (RISC-V Vector) instructions marked as RoCC
 *
 * This decoder takes standard RVV instructions and marks them as RoCC instructions,
 * allowing them to be handled by a RoCC accelerator instead of the built-in vector unit.
 *
 * Usage: Enable by setting usingRVVRoCC parameter to true in RocketTileParamsBB
 * Note: usingVector and usingRVVRoCC cannot both be true simultaneously
 *
 * This decoder uses opcode-based matching patterns instead of listing all 447 RVV instructions
 * individually to avoid JVM method size limits and memory issues during decode minimization.
 *
 * RVV instruction opcodes:
 * - 1010111 (0x57): V-type arithmetic/logic instructions, vector config
 * - 0000111 (0x07): Vector load instructions (funct3 != 010)
 * - 0100111 (0x27): Vector store instructions (funct3 != 010)
 *
 * Note: This decoder uses wildcard patterns that may match invalid encodings.
 * The RoCC accelerator should validate instructions and handle illegal opcodes appropriately.
 */
class RVVRoCCDecode(implicit val p: Parameters) extends DecodeConstants with ScalarOpConstants with MemoryOpConstants
{
  import freechips.rocketchip.rocket.Instructions._

  // Common decode signals for RVV instructions (rocc field set to Y)
  // Most RVV instructions don't read scalar registers by default (overridden by specific patterns)
  private val rvvRoCCBase: List[BitPat] = List(Y,N,Y,N,N,N,N,N,A2_X,   A1_X,   IMM_X, DW_X,  FN_X,     N,M_X,        N,N,N,N,N,N,N,CSR.N,N,N,N,N)

  // Vector configuration instructions need wxd=Y and may read rs1
  private val vcfgSigs: List[BitPat]     = List(Y,N,Y,N,N,N,N,Y,A2_X,   A1_X,   IMM_X, DW_X,  FN_X,     N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N)
  private val vcfgImmSigs: List[BitPat]  = List(Y,N,Y,N,N,N,N,N,A2_X,   A1_X,   IMM_X, DW_X,  FN_X,     N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N)

  val table: Array[(BitPat, List[BitPat])] = Array(
    // All V-type instructions (opcode = 1010111)
    // This includes VSETVL, VSETVLI, VSETIVLI and all arithmetic/logic vector instructions
    // Note: VSET* instructions should ideally have wxd=Y and rxs1=Y, but we use uniform
    // decode signals here to avoid pattern overlap issues. The RoCC accelerator should
    // handle the detailed decoding of vset instructions.
    BitPat("b?????????????????????????????????1010111") -> rvvRoCCBase,

    // Vector load instructions (opcode = 0000111, funct3 = 000/101/110/111)
    // funct3=101: VLE16, VLE256, etc.
    // funct3=110: VLE32, VLE512, etc.
    // funct3=111: VLE64, VLE1024, etc.
    BitPat("b?????????????????000?????0000111") -> rvvRoCCBase,
    BitPat("b?????????????????101?????0000111") -> rvvRoCCBase,
    BitPat("b?????????????????110?????0000111") -> rvvRoCCBase,
    BitPat("b?????????????????111?????0000111") -> rvvRoCCBase,

    // Vector store instructions (opcode = 0100111, but NOT funct3=010 which is FP store)
    // funct3=000: VSE8, VSE128, VS*R, etc.
    // funct3=101: VSE16, VSE256, etc.
    // funct3=110: VSE32, VSE512, etc.
    // funct3=111: VSE64, VSE1024, etc.
    BitPat("b?????????????????000?????0100111") -> rvvRoCCBase,
    BitPat("b?????????????????101?????0100111") -> rvvRoCCBase,
    BitPat("b?????????????????110?????0100111") -> rvvRoCCBase,
    BitPat("b?????????????????111?????0100111") -> rvvRoCCBase
  )
}
