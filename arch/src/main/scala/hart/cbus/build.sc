import mill._
import $file.^.^.^.^.^.settings

val ipRoot: os.Path = millSourcePath
trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass = Some("framework.hart.cbus.Emit")
}

// The registry is compiled with its consumers; framework never imports component implementations.
def registry(sources: Seq[PathRef], destination: os.Path): Seq[PathRef] = {
  val files     = sources.flatMap { source =>
    if (os.isDir(source.path))
      os.walk(source.path).filter(p => p.ext == "scala" && !p.segments.exists(Set("out", "generated", "target", "build")))
    else Seq(source.path)
  }
  val factories = files.flatMap { source =>
    val text  = os.read(source)
    val names = "(?m)^object (\\w+) extends HartFactory".r.findAllMatchIn(text).map(_.group(1)).toSeq
    if (names.isEmpty) Seq.empty
    else {
      val pkg = "(?m)^package ([\\w.]+)".r.findFirstMatchIn(text).get.group(1)
      names.map(name => s"$pkg.$name")
    }
  }.distinct.sorted
  val output    = destination / "HartComponents.scala"
  os.write.over(
    output,
    s"""package framework.hart.cbus.generated
import framework.hart.cbus.HartFactory
object HartComponents {
  val components: Seq[HartFactory] = Seq(${factories.mkString(",\n")})
  require(components.map(_.component).distinct.size == components.size, "Duplicate Hart component registration")
  val factories: Map[String, HartFactory] = components.map(f => f.component -> f).toMap
}
"""
  )
  Seq(PathRef(output))
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("CBus.scala"))
  )
)

def emitSources = T.sources(
  Seq.empty[PathRef]
)
