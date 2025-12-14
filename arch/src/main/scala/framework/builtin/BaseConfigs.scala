package framework.builtin

import scala.math.{max, pow, sqrt}
import chisel3._
import chisel3.util._
import freechips.rocketchip.tile.OpcodeSet
import org.chipsalliance.cde.config._

import framework.memdomain.dma.LocalAddr

sealed abstract trait BuckyballMemCapacity
case class CapacityInKilobytes(kilobytes: Int) extends BuckyballMemCapacity
case class CapacityInVectors(vectors: Int) extends BuckyballMemCapacity
// case class CapacityInMatrices(matrices: Int) extends BuckyballMemCapacity

case class BaseConfig(
  opcodes: OpcodeSet = OpcodeSet.custom3,

  inputType: Data,
  accType: Data,

  veclane: Int = 16,
  accveclane: Int = 4,

  tlb_size: Int = 4,
  // Number of RoB entries
  rob_entries: Int = 16,
  // Whether reservation station responds out-of-order (false = wait for ROB to be empty before responding)
  rs_out_of_order_response: Boolean = true,

  // Unused
  dma_maxbytes: Int = 64,
  dma_buswidth: Int = 128,

  sp_banks: Int = 4,
  acc_banks: Int = 8,

  sp_singleported: Boolean = true,

  sp_capacity: BuckyballMemCapacity = CapacityInKilobytes(256),
  acc_capacity: BuckyballMemCapacity = CapacityInKilobytes(64),

  max_in_flight_mem_reqs: Int = 16, // Unused
  aligned_to: Int = 1,
  spad_read_delay: Int = 0,

  // Index length supporting SPAD (16384 rows) + ACC (4096 rows)
  spAddrLen: Int = 15,
  // Index length for 4GB
  memAddrLen: Int = 32,

  // Number of vector PEs per thread
  numVecPE: Int = 16,
  // Number of vector threads per thread
  numVecThread: Int = 16,

  // Empty ball id
  emptyBallid: Int = 5,

) {
  val spad_w = veclane * inputType.getWidth
  val spad_mask_len = (spad_w / (aligned_to * 8)) max 1
  val spad_bank_entries = sp_capacity match {
    case CapacityInKilobytes(kb) => kb * 1024 * 8 / (sp_banks * spad_w)
    case CapacityInVectors(vs) => vs * veclane / sp_banks
  }
  val acc_w = accveclane * accType.getWidth
  val acc_mask_len = (acc_w / (aligned_to * 8)) max 1
  val acc_bank_entries = acc_capacity match {
    case CapacityInKilobytes(kb) => kb * 1024 * 8 / (acc_banks * acc_w)
    case CapacityInVectors(vs) => vs * accveclane / acc_banks
  }


  val local_addr_t = new LocalAddr(sp_banks, spad_bank_entries, acc_banks, acc_bank_entries)



}
