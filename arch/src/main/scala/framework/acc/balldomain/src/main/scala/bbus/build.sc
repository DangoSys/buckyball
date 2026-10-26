import mill._

// The component library owns the generated table; framework only knows the contract.
def registry(sources: Seq[PathRef], componentRoot: os.Path, destination: os.Path): Seq[PathRef] = {
  val declarations = sources.flatMap { source =>
    val text  = os.read(source.path)
    val names = "(?m)^object (\\w+) extends BallFactory".r.findAllMatchIn(text).map(_.group(1)).toSeq
    if (names.isEmpty) Seq.empty
    else {
      val pkg       = "(?m)^package ([\\w.]+)".r.findFirstMatchIn(text).get.group(1)
      val component = source.path.relativeTo(componentRoot).segments.head
      names.map(name => component -> s"$pkg.$name")
    }
  }.sortBy(_._1)
  require(declarations.map(_._1).distinct.size == declarations.size, "Duplicate component factory")
  val entries      = declarations.map { case (component, factory) => s""""$component" -> $factory""" }
  val output       = destination / "BallComponents.scala"
  os.write.over(
    output,
    s"""package framework.acc.balldomain.bbus.generated
import framework.acc.balldomain.bbus.BallFactory
object BallComponents {
  val factories: Map[String, BallFactory] = Map(${entries.mkString(",\n")})
}
"""
  )
  Seq(PathRef(output))
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("bbus.scala")),
    PathRef(millSourcePath / os.RelPath("channels/Channels.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
