import mill._
import $file.thirdparty.{build => external}
import $file.src.main.scala.{build => framework_build}
import $file.^.examples.{build => example_build}

object architecture extends Module {
  def thirdparty = external
  def framework  = framework_build
  def examples   = example_build
}
