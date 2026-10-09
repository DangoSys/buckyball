ThisBuild / scalaVersion := "2.13.16"
ThisBuild / organization := "srtp.fpga"
ThisBuild / version := "0.1.0"
ThisBuild / scalacOptions ++= Seq("-language:reflectiveCalls", "-Ymacro-annotations", "-deprecation", "-feature")

lazy val chiselSettings = Seq(
  scalaVersion := "2.13.16",
  libraryDependencies += "org.chipsalliance" %% "chisel" % "6.7.0",
  addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % "6.7.0" cross CrossVersion.full)
)
lazy val cde = project.in(file("thirdparty/cde")).settings(chiselSettings).settings(
  Compile / unmanagedSourceDirectories += baseDirectory.value / "cde/src/chipsalliance"
)
lazy val diplomacy = project.in(file("thirdparty/diplomacy/diplomacy")).dependsOn(cde).settings(chiselSettings).settings(
  Compile / unmanagedSourceDirectories += baseDirectory.value / "src/diplomacy",
  libraryDependencies += "com.lihaoyi" %% "sourcecode" % "0.3.0"
)
lazy val hardfloat = project.in(file("thirdparty/hardfloat/hardfloat")).dependsOn(cde).settings(chiselSettings)
lazy val macros = project.in(file("thirdparty/rocket-chip/macros")).settings(
  libraryDependencies += "org.scala-lang" % "scala-reflect" % "2.13.16"
)
lazy val rocket = project.in(file("thirdparty/rocket-chip")).dependsOn(cde, diplomacy, hardfloat, macros).settings(chiselSettings).settings(
  libraryDependencies ++= Seq(
    "com.lihaoyi" %% "mainargs" % "0.5.0",
    "org.json4s" %% "json4s-jackson" % "4.0.5",
    "org.scala-graph" %% "graph-core" % "1.13.5"
  )
)
lazy val inclusive = project.in(file("thirdparty/inclusive-cache")).dependsOn(rocket).settings(chiselSettings).settings(
  Compile / unmanagedSourceDirectories += baseDirectory.value / "design/craft"
)
lazy val root = project.in(file(".")).dependsOn(inclusive).settings(chiselSettings).settings(
  name := "pebble-arty-a7",
  libraryDependencies ++= Seq(
    "com.lihaoyi" %% "upickle" % "3.3.1",
    "tech.sparse" %% "toml-scala" % "0.2.2"
  ),
  Compile / unmanagedSources ++= {
    val upstream = baseDirectory.value / "upstream"
    val framework = upstream / "arch/src/main/scala/framework"
    val directories = Seq("balldomain", "dpi", "frontend", "gpdomain", "memdomain", "top",
      "system/core/rocket", "system/core/accelerator", "system/tile")
    val shared = directories.flatMap(d => ((framework / d) ** "*.scala").get)
      .filterNot(f => Set("WithBuckyballTiles.scala", "WithBoomTile.scala")(f.getName))
    shared ++ ((upstream / "arch/src/main/scala/sims/hash") ** "*.scala").get ++ Seq("smatmul", "matadd", "relu").flatMap { name =>
      val ball = upstream / "examples/balls" / name
      ((ball / "arch/src/main/scala") ** "*.scala").get ++ ((ball / "configs") ** "*.scala").get
    }
  }
)
