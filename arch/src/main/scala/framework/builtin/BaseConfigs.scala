package framework.builtin

import scala.math.{max, pow, sqrt}
import chisel3._
import chisel3.util._
import freechips.rocketchip.tile._
import org.chipsalliance.cde.config._

import framework.builtin.memdomain.dma.LocalAddr

sealed abstract trait BuckyBallMemCapacity
case class CapacityInKilobytes(kilobytes: Int) extends BuckyBallMemCapacity
case class CapacityInVectors(vectors: Int) extends BuckyBallMemCapacity
// case class CapacityInMatrices(matrices: Int) extends BuckyBallMemCapacity

case class BaseConfig(
  opcodes: OpcodeSet = OpcodeSet.custom3,
  
  inputType: Data,
  accType: Data,

  veclane: Int = 16,
  accveclane: Int = 4, 

  tlb_size: Int = 4,
  rob_entries: Int = 16,  // RoB条目数量
  
  dma_maxbytes: Int = 64, // 未使用
  dma_buswidth: Int = 128,
  
  sp_banks: Int = 2,
  acc_banks: Int = 4,
  
  sp_singleported: Boolean = true,
  
  sp_capacity: BuckyBallMemCapacity = CapacityInKilobytes(16),
  acc_capacity: BuckyBallMemCapacity = CapacityInKilobytes(8),
  
  max_in_flight_mem_reqs: Int = 16, // 未使用
  aligned_to: Int = 1, 
  spad_read_delay: Int = 0,

  spAddrLen: Int = 14, // 256KB的索引长度
  memAddrLen: Int = 32, // 4GB的索引长度

  numVecPE: Int = 16, // 每个线程的向量PE数量
  numVecThread: Int = 16, // 每个线程的向量线程数量
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


// object BuckyBallConfigs {
//   val defaultConfig = BuckyBallConfig(
//     inputType = UInt(8.W),
//     accType = UInt(32.W)
//   )
// }