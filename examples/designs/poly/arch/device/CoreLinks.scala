package examples.designs.poly

import framework.device.shm.{Params => CommunicationParams, Link, Router}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.ip.bus.axi4

/**
 * The core links: a 3x3 mesh of routers, a router per core, cut down to the transfers of poly's device programs.
 *
 * Those only go between the reduce core (core 0) and a poly core: core 0 hands the ranks their inputs and takes
 * their results. Core 0 sits in the middle of the mesh and the poly cores go round its edge:
 * {{{
 *   8 7 6
 *   1 0 5
 *   2 3 4
 * }}}
 * The mesh keeps a link from core 0 to each of its neighbours 1, 3, 5, 7 and a link from each of those to the next
 * core round the edge, 2, 4, 6, 8: a tree with core 0 at its root, so core 0 is one hop from 1, 3, 5, 7 and two
 * from 2, 4, 6, 8, and no other link exists. Two poly cores have no route between them.
 */
@instantiable
class CoreLinks(parameters: CommunicationParams) extends Module {
  val p     = parameters.axi
  val count = parameters.cores
  require(count >= 2 && count <= 9)

  // The cores a write from `core` to core 0 goes through.
  def toReduce(core: Int): Seq[Int] = if (core == 0) Seq(0) else core +: toReduce(if (core % 2 == 1) 0 else core - 1)
  // (source, target, the cores on the way)
  val paths = (1 until count).flatMap(core => Seq((core, 0, toReduce(core)), (0, core, toReduce(core).reverse)))
  val hops  = paths.flatMap { case (_, _, nodes) => nodes.zip(nodes.tail) }.distinct

  val sharedBytes = parameters.capacity
  require(p.dataBits == 128 && sharedBytes >= p.bytes)
  val localBits   = log2Ceil(sharedBytes)
  require(p.addressBits >= localBits + log2Ceil(count))

  @public
  val io = IO(new Bundle {
    val cores = Vec(count, Flipped(new Link(p)))
  })

  // The ports of the router of a core: the core's tx and rx first, then a port per link.
  def ins(core:  Int) = None +: hops.filter(_._2 == core).map(hop => Some(hop._1))
  def outs(core: Int) = None +: hops.filter(_._1 == core).map(hop => Some(hop._2))

  val routers: Seq[Instance[Router]] = (0 until count).map { core =>
    // The targets an input sends to an output: the paths that enter by the input and leave by the output.
    val routes = ins(core).map { in =>
      outs(core).map { out =>
        paths.filter { case (_, _, nodes) =>
          val at = nodes.indexOf(core)
          at >= 0 && (if (at == 0) in.isEmpty else in.contains(nodes(at - 1))) &&
          (if (at == nodes.size - 1) out.isEmpty else out.contains(nodes(at + 1)))
        }.map(_._2)
      }
    }
    Instantiate(new Router(p, localBits, routes))
  }

  for (core <- 0 until count) {
    val link = io.cores(core)
    routers(core).io.in(0) <> link.tx
    link.rx <> routers(core).io.out(0)
    link.rx.aw.bits.addr := routers(core).io.out(0).aw.bits.addr(localBits - 1, 0)
    for ((Some(next), output) <- outs(core).zipWithIndex)
      routers(next).io.in(ins(next).indexOf(Some(core))) <> routers(core).io.out(output)

    def routed(target: UInt) = paths.filter(_._1 == core).map(path => target === path._2.U).reduce(_ || _)
    link.routeAvailable := routed(link.routeTarget)

    val address = link.tx.aw.bits
    val offset  = address.addr(localBits - 1, 0)
    val bytes   = (address.len +& 1.U) << log2Ceil(p.bytes)
    when(link.tx.aw.valid) {
      assert(routed(address.addr >> localBits), "CLink address selects a core without a route")
      assert(
        address.size === log2Ceil(p.bytes).U && address.burst === 1.U && !address.lock,
        "CLink requires full-width INCR bursts"
      )
      assert(
        offset(log2Ceil(p.bytes) - 1, 0) === 0.U && (offset +& bytes) <= sharedBytes.U,
        "CLink burst exceeds shared storage or is unaligned"
      )
      assert((offset.pad(12)(11, 0) +& bytes) <= 4096.U, "CLink burst crosses a 4KiB boundary")
    }
  }
}
