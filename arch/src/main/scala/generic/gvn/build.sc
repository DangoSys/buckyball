import mill._
import $file.^.^.^.^.^.settings
import $file.^.^.^.^.^.thirdparty.{build => thirdparty_build}

val ipRoot: os.Path = millSourcePath
val generic         = millSourcePath / os.up

object library extends settings.FrameworkModule {
  override def moduleRoot = ipRoot
  override def moduleDeps = Seq(thirdparty_build.protoJava)
  override def sources    = T.sources(
    Seq(
      PathRef(ipRoot / "src" / "main" / "scala" / "Params.scala"),
      PathRef(ipRoot / "src" / "main" / "scala" / "CheckerOnChip.scala"),
      PathRef(generic / "axi4" / "src" / "main" / "scala" / "Interface.scala"),
      PathRef(generic / "axi4" / "src" / "main" / "scala" / "Params.scala")
    )
  )
}

object emit extends settings.IpModule {
  override def moduleRoot = ipRoot
  override def framework  = library
  override def mainClass  = Some("framework.generic.gvn.Emit")
}
