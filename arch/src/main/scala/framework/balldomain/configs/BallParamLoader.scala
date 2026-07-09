package framework.balldomain.configs

import framework.top.GlobalConfig
import toml.{Toml, Value}

object BallParamLoader {

  def load(path: String): Map[String, Value] = {
    val content = scala.io.Source.fromFile(path, "UTF-8").mkString
    Toml.parse(content) match {
      case Right(Value.Tbl(root)) => root
      case Right(_)               =>
        throw new RuntimeException(s"TOML root must be a table in $path")
      case Left((addr, msg))      =>
        throw new RuntimeException(s"TOML parse error at $addr: $msg in $path")
    }
  }

  def ball(root: Map[String, Value]): Map[String, Value] =
    root.get("ball") match {
      case Some(Value.Tbl(t)) => t
      case None               => throw new RuntimeException("Missing TOML section [ball]")
      case _                  => throw new RuntimeException("Expected TOML section [ball]")
    }

  def ballDir(ballClass: String): String = {
    val parts = ballClass.split("\\.")
    if (parts.length < 4 || parts(0) != "examples" || parts(1) != "balls")
      throw new RuntimeException(s"Invalid ballClass for ballDir: $ballClass")
    s"../${parts(0)}/${parts(1)}/${parts(2)}"
  }

  def ballTable(b: GlobalConfig, ballName: String): Map[String, Value] = {
    val mapping = b.ballDomain.ballIdMappings.find(_.ballName == ballName) match {
      case Some(m) => m
      case None    => throw new RuntimeException(s"No ballIdMapping for ballName=$ballName")
    }
    val path    = s"${ballDir(mapping.ballClass)}/${mapping.config}"
    ball(load(path))
  }

  def int(table: Map[String, Value], key: String): Int =
    table.get(key) match {
      case Some(Value.Num(n)) => n.toInt
      case None               => throw new RuntimeException(s"Missing integer at key '$key'")
      case _                  => throw new RuntimeException(s"Expected integer at key '$key'")
    }

  def str(table: Map[String, Value], key: String): String =
    table.get(key) match {
      case Some(Value.Str(s)) => s
      case None               => throw new RuntimeException(s"Missing string at key '$key'")
      case _                  => throw new RuntimeException(s"Expected string at key '$key'")
    }

  def bool(table: Map[String, Value], key: String): Boolean =
    table.get(key) match {
      case Some(Value.Bool(b)) => b
      case None                => throw new RuntimeException(s"Missing boolean at key '$key'")
      case _                   => throw new RuntimeException(s"Expected boolean at key '$key'")
    }

}
