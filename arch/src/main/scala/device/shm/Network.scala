package framework.device.shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

/**
 * The network of the device SHM nodes: a router per hart and the links a design keeps between them.
 *
 * `links` are the kept links of the design's topology (a mesh pruned to its dataflow); they must form a tree over the
 * harts. A write goes from its source to its target hart along the one path of the tree between them: up towards
 * the root to the first hart the two have in common, then down. The routers forward it hop by hop, so every hart
 * reaches every other one and the software issues one transfer per (value, target). A tree has no cycle of channel
 * dependencies (a write turns from up to down at most once), so the circuit-switched routers cannot deadlock.
 */
@instantiable
class Network(parameters: Params, links: Seq[(Int, Int)]) extends Module {
  val p     = parameters.axi
  val count = parameters.harts
  require(count >= 2 && links.size == count - 1)
  require(links.forall { case (a, b) => a != b && a >= 0 && b >= 0 && a < count && b < count })

  def neighbours(hart: Int): Seq[Int] = links.collect { case (a, b) if a == hart => b; case (a, b) if b == hart => a }

  // The path of the tree from `from` to `to`, both ends included.
  def route(from: Int, to: Int): Seq[Int] = {
    def search(path: Seq[Int]): Option[Seq[Int]] =
      if (path.last == to) Some(path)
      else neighbours(path.last).filterNot(path.contains).view.flatMap(next => search(path :+ next)).headOption
    search(Seq(from)).getOrElse(throw new IllegalArgumentException(s"The links do not connect hart $from to hart $to"))
  }

  // (source, target, the harts on the way)
  val paths = for {
    from <- 0 until count
    to   <- 0 until count if from != to
  } yield (from, to, route(from, to))

  val hops = links.flatMap { case (a, b) => Seq((a, b), (b, a)) }

  val sharedBytes = parameters.capacity
  require(p.dataBits == 128 && sharedBytes >= p.bytes)
  val localBits   = log2Ceil(sharedBytes)
  require(p.addressBits >= localBits + log2Ceil(count))

  @public
  val io = IO(new Bundle {
    val harts = Vec(count, Flipped(new Link(p)))
  })

  // The ports of the router of a hart: the hart's tx and rx first, then a port per link.
  def ins(hart:  Int) = None +: hops.filter(_._2 == hart).map(hop => Some(hop._1))
  def outs(hart: Int) = None +: hops.filter(_._1 == hart).map(hop => Some(hop._2))

  val routers: Seq[Instance[Router]] = (0 until count).map { hart =>
    // The targets an input sends to an output: the paths that enter by the input and leave by the output.
    val routes = ins(hart).map { in =>
      outs(hart).map { out =>
        paths.filter { case (_, _, nodes) =>
          val at = nodes.indexOf(hart)
          at >= 0 && (if (at == 0) in.isEmpty else in.contains(nodes(at - 1))) &&
          (if (at == nodes.size - 1) out.isEmpty else out.contains(nodes(at + 1)))
        }.map(_._2)
      }
    }
    Instantiate(new Router(p, localBits, routes))
  }

  for (hart <- 0 until count) {
    val link = io.harts(hart)
    routers(hart).io.in(0) <> link.tx
    link.rx <> routers(hart).io.out(0)
    link.rx.aw.bits.addr := routers(hart).io.out(0).aw.bits.addr(localBits - 1, 0)
    for ((Some(next), output) <- outs(hart).zipWithIndex)
      routers(next).io.in(ins(next).indexOf(Some(hart))) <> routers(hart).io.out(output)

    def routed(target: UInt) = target =/= hart.U && target < count.U
    link.routeAvailable := routed(link.routeTarget)

    val address = link.tx.aw.bits
    val offset  = address.addr(localBits - 1, 0)
    val bytes   = (address.len +& 1.U) << log2Ceil(p.bytes)
    when(link.tx.aw.valid) {
      assert(routed(address.addr >> localBits), "HLink address selects no other hart")
      assert(
        address.size === log2Ceil(p.bytes).U && address.burst === 1.U && !address.lock,
        "HLink requires full-width INCR bursts"
      )
      assert(
        offset(log2Ceil(p.bytes) - 1, 0) === 0.U && (offset +& bytes) <= sharedBytes.U,
        "HLink burst exceeds shared storage or is unaligned"
      )
      assert((offset.pad(12)(11, 0) +& bytes) <= 4096.U, "HLink burst crosses a 4KiB boundary")
    }
  }
}
