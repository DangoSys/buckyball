import mill._
import $file.src.main.scala.router.{build => router_build}
import $file.^.^.^.^.^.^.settings

val ipRoot: os.Path = millSourcePath

trait Emitter extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def mainClass  = Some("framework.core.tbus.Emit")
}

// The component library owns the generated table; framework only knows the contract.
def registry(sources: Seq[PathRef], componentRoot: os.Path, destination: os.Path): Seq[PathRef] = {
  val declarations = sources.flatMap { source =>
    val text  = os.read(source.path)
    val names = "(?m)^object (\\w+) extends ThreadFactory".r.findAllMatchIn(text).map(_.group(1)).toSeq
    if (names.isEmpty) Seq.empty
    else {
      val pkg       = "(?m)^package ([\\w.]+)".r.findFirstMatchIn(text).get.group(1)
      val component = source.path.relativeTo(componentRoot).segments.head
      names.map(name => component -> s"$pkg.$name")
    }
  }.sortBy(_._1)
  require(declarations.map(_._1).distinct.size == declarations.size, "Duplicate component factory")
  val entries      = declarations.map { case (component, factory) => s""""$component" -> $factory""" }
  val output       = destination / "ThreadComponents.scala"
  os.write.over(
    output,
    s"""package framework.core.tbus.generated
import framework.core.tbus.ThreadFactory
object ThreadComponents {
  val factories: Map[String, ThreadFactory] = Map(${entries.mkString(",\n")})
}
"""
  )
  Seq(PathRef(output))
}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/SharedNetwork.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/TBus.scala")),
    PathRef(millSourcePath / os.RelPath("src/main/scala/configs/Params.scala"))
  ) ++
  router_build.sources()
)

def emitSources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("src/main/scala/Emit.scala"))
  ) ++
  router_build.emitSources()
)
