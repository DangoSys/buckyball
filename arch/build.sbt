// See README.md for license details.

val chisel6Version = "6.5.0"
val chiselTestVersion = "6.0.0"
val scalaVersionFromChisel = "2.13.12"

// Fix for scalafix undefined setting
ThisBuild / scalafixScalaBinaryVersion := scalaBinaryVersion.value

// ------------------------------------------------------------------------------
// Chisel Version Config
// ------------------------------------------------------------------------------

lazy val chisel6Settings = Seq(
  libraryDependencies ++= Seq("org.chipsalliance" %% "chisel" % chisel6Version),
  addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chisel6Version cross CrossVersion.full)
)

lazy val chiselSettings = chisel6Settings ++ Seq(
  libraryDependencies ++= Seq(
    "org.apache.commons" % "commons-lang3" % "3.12.0",
    "org.apache.commons" % "commons-text" % "1.9"
  )
)

lazy val scalaTestSettings =  Seq(
  libraryDependencies ++= Seq(
    "org.scalatest" %% "scalatest" % "3.2.+" % "test"
  )
)

// ------------------------------------------------------------------------------
// Dependencies
// ------------------------------------------------------------------------------
lazy val chipyard = ProjectRef(file("thirdparty/chipyard"), "chipyard")
lazy val firechip = ProjectRef(file("thirdparty/chipyard"), "firechip")

// Palladium FPGA subproject (external reference)
lazy val palladium = ProjectRef(file("../tools/palladium"), "palladium")

// ------------------------------------------------------------------------------
// Project Settings
// ------------------------------------------------------------------------------
lazy val buckyball = (project in file("."))
  .dependsOn(chipyard, firechip, palladium)
  .settings(
    name := "buckyball",
    organization := "com.buckyball",
    version := "1.0.0",
    scalaVersion := scalaVersionFromChisel,
    scalacOptions ++= Seq(
      "-deprecation",
      "-unchecked",
      "-Ymacro-annotations"
    ),
    resolvers ++= Seq(
      Resolver.sonatypeRepo("snapshots"),
      Resolver.sonatypeRepo("releases")
    ),
    chisel6Settings ++
    scalaTestSettings ++
    Seq(
      libraryDependencies ++= Seq(
        "edu.berkeley.cs" %% "rocketchip" % "1.6"
      )
    )
  )
