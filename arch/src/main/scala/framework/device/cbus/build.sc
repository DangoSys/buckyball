import mill._

// The registry is compiled with its consumers; framework never imports component implementations.
def registry(sources: Seq[PathRef], destination: os.Path): Seq[PathRef] = {
  val files     = sources.flatMap { source =>
    if (os.isDir(source.path))
      os.walk(source.path).filter(p => p.ext == "scala" && !p.segments.exists(Set("out", "generated", "target", "build")))
    else Seq(source.path)
  }
  val factories = files.flatMap { source =>
    val text  = os.read(source)
    val names = "(?m)^object (\\w+) extends CoreFactory".r.findAllMatchIn(text).map(_.group(1)).toSeq
    if (names.isEmpty) Seq.empty
    else {
      val pkg = "(?m)^package ([\\w.]+)".r.findFirstMatchIn(text).get.group(1)
      names.map(name => s"$pkg.$name")
    }
  }.distinct.sorted
  val output    = destination / "CoreComponents.scala"
  os.write.over(
    output,
    s"""package framework.device.cbus.generated
import framework.device.cbus.CoreFactory
object CoreComponents {
  val components: Seq[CoreFactory] = Seq(${factories.mkString(",\n")})
  require(components.map(_.component).distinct.size == components.size, "Duplicate Core component registration")
  val factories: Map[String, CoreFactory] = components.map(f => f.component -> f).toMap
}
"""
  )
  Seq(PathRef(output))
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("C2C.scala")),
    PathRef(millSourcePath / os.RelPath("CBus.scala")),
    PathRef(millSourcePath / os.RelPath("Control.scala")),
    PathRef(millSourcePath / os.RelPath("Fabric.scala")),
    PathRef(millSourcePath / os.RelPath("SharedAxiEndpoint.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
