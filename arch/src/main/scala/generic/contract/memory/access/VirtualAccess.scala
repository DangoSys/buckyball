package framework.generic.contract.memory.access

import framework.generic.contract.memory.access.PhysicalRegion

import buckyball.config.RegionHardware
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.bus.chi.rnf.{CacheAccess, CacheAtomic, CacheResult}

case class PhysicalRegion(
  base:       BigInt,
  bytes:      BigInt,
  cacheable:  Boolean,
  executable: Boolean,
  readable:   Boolean,
  writable:   Boolean,
  atomic:     Boolean,
  normal:     Boolean)

object PhysicalRegion {

  def fromProto(p: RegionHardware): PhysicalRegion = PhysicalRegion(
    BigInt(java.lang.Long.toUnsignedString(p.getBase)),
    BigInt(java.lang.Long.toUnsignedString(p.getBytes)),
    p.getCacheable,
    p.getExecutable,
    p.getReadable,
    p.getWritable,
    p.getAtomic,
    p.getNormal
  )

}
