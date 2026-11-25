package examples.designs.poly.devices.poly

/**
 * The hart links: a 3x3 mesh with a router per hart, cut down to the transfers of poly's device programs.
 *
 * Those go between the reduce hart (hart 0) and a poly hart: hart 0 hands the ranks their inputs and takes their
 * results. Hart 0 sits in the middle of the mesh and the poly harts go round its edge:
 * {{{
 *   8 7 6
 *   1 0 5
 *   2 3 4
 * }}}
 * The mesh keeps a link from hart 0 to each of its neighbours 1, 3, 5, 7 and a link from each of those to the next
 * hart round the edge, 2, 4, 6, 8: a tree with hart 0 at its root, so hart 0 is one hop from 1, 3, 5, 7 and two
 * from 2, 4, 6, 8. The routers of the device SHM network (framework.device.shm.Network) route every pair of harts
 * over the tree: a write between two poly harts goes up to their common hart and down, at most four hops, routed by
 * the hardware.
 */
object HartLinks {
  // The (column, row) of each hart in the mesh.
  val positions = Seq((1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0))
  val kept      = Seq((0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6), (0, 7), (7, 8))
  require(kept.forall { case (a, b) =>
    (positions(a)._1 - positions(b)._1).abs + (positions(a)._2 - positions(b)._2).abs == 1
  })

  // The kept links between the first `count` harts.
  def apply(count: Int): Seq[(Int, Int)] = {
    require(count >= 2 && count <= positions.size)
    kept.filter { case (a, b) => a < count && b < count }
  }

}
