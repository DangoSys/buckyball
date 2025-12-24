package framework.builtin

import scala.math.{max, pow, sqrt}
import chisel3._
import chisel3.util._
import chisel3.experimental.SerializableModuleParameter
import freechips.rocketchip.tile.OpcodeSet
import org.chipsalliance.cde.config._

sealed abstract trait BuckyballMemCapacity
case class CapacityInKilobytes(kilobytes: Int) extends BuckyballMemCapacity
case class CapacityInVectors(vectors: Int) extends BuckyballMemCapacity
// case class CapacityInMatrices(matrices: Int) extends BuckyballMemCapacity

object BaseConfig {

  // upickle support for BuckyballMemCapacity
  implicit def memCapacityRW: upickle.default.ReadWriter[BuckyballMemCapacity] =
    upickle.default.ReadWriter.merge(
      upickle.default.macroRW[CapacityInKilobytes],
      upickle.default.macroRW[CapacityInVectors]
    )

  // upickle support for BaseConfig
  implicit def rw: upickle.default.ReadWriter[BaseConfig] = upickle.default.macroRW
}

case class BaseConfig(
  tlb_size:                 Int = 4,
  // Number of RoB entries
  rob_entries:              Int = 16,
  // Whether reservation station responds out-of-order (false = wait for ROB to be empty before responding)
  rs_out_of_order_response: Boolean = true,
  // Unused
  dma_maxbytes:             Int = 64,
  dma_buswidth:             Int = 128,
  bankNum:                  Int = 32,
  // bit
  bankWidth:                Int = 128,
  bankCapacity:             BuckyballMemCapacity = CapacityInKilobytes(16),
  sp_singleported:          Boolean = true,
  max_in_flight_mem_reqs:   Int = 16, // Unused
  aligned_to:               Int = 1,
  spad_read_delay:          Int = 0,
  // Index length supporting SPAD (16384 rows) + ACC (4096 rows)
  // spAddrLen: Int = 15,
  // Index length for 4GB
  memAddrLen:               Int = 32,
  // Number of vector PEs per thread
  numVecPE:                 Int = 16,
  // Number of vector threads per thread
  numVecThread:             Int = 16,
  // Empty ball id
  emptyBallid:              Int = 5,
  // Bank channel (BBus bandwidth, same as MemDomain bankChannel)
  bankChannel:              Int = 8)
    extends SerializableModuleParameter {
  // Fixed data widths
  val inputWidth: Int = 8  // UInt8
  val accWidth:   Int = 32 // UInt32

  val bankMaskLen = (bankWidth / (aligned_to * 8)) max 1

  val bankEntries = bankCapacity match {
    case CapacityInKilobytes(kb) => kb * 1024 * 8 / (bankNum * bankWidth)
  }

  // Helper methods to get Data types
  def inputType: Data = UInt(inputWidth.W)
  def accType:   Data = UInt(accWidth.W)

  override def toString: String =
    s"""BuckyballConfig
       |  ROB entries: $rob_entries
       |  Bank num: $bankNum
       |  Bank width: $bankWidth bits
       |  Bank entries: $bankEntries
       |  Bank capacity: $bankCapacity
       |  TLB size: $tlb_size
       |  DMA max bytes: $dma_maxbytes
       |""".stripMargin
}
