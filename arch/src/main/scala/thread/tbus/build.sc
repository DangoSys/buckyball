import mill._

// The component library owns the generated table; framework only knows the contract.
def registry(
  sources:       Seq[PathRef],
  componentRoot: os.Path,
  destination:   os.Path,
  namespace:     String
): Seq[PathRef] = {
  val files = sources.flatMap { source =>
    if (os.isDir(source.path))
      os.walk(source.path).filter(p => p.ext == "scala" && !p.segments.exists(Set("out", "generated", "target", "build")))
    else Seq(source.path)
  }
  val declarations = files.flatMap { source =>
    val text  = os.read(source)
    val names = "(?m)^object (\\w+) extends ThreadFactory".r.findAllMatchIn(text).map(_.group(1)).toSeq
    if (names.isEmpty) Seq.empty
    else {
      val pkg       = "(?m)^package ([\\w.]+)".r.findFirstMatchIn(text).get.group(1)
      val component = source.relativeTo(componentRoot).segments.head
      names.map(name => component -> s"$pkg.$name")
    }
  }.sortBy(_._1)
  require(declarations.map(_._1).distinct.size == declarations.size, "Duplicate component factory")
  val entries      = declarations.map { case (component, factory) => s""""$component" -> $factory""" }
  val output       = destination / "ThreadComponents.scala"
  os.write.over(
    output,
    s"""package $namespace
import framework.thread.tbus.ThreadFactory
object ThreadComponents {
  val factories: Map[String, ThreadFactory] = Map(${entries.mkString(",\n")})
}
"""
  )
  Seq(PathRef(output))
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("TBus.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
