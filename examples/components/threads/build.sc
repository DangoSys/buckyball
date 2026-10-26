import mill._

def sources = T.sources {
  os.list(millSourcePath).filter(os.isDir).flatMap { root =>
    val source  = root / "arch"
    val configs = root / "configs"
    (if (os.exists(source)) os.walk(source).filter(p => p.ext == "scala" && !p.toString.contains("/out/") && !p.toString.contains("/build/")) else Seq.empty) ++
      (if (os.exists(configs)) os.walk(configs).filter(_.ext == "scala") else Seq.empty)
  }.map(PathRef(_))
}
