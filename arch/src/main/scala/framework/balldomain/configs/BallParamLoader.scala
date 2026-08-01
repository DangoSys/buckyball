package framework.balldomain.configs

import framework.top.GlobalConfig
import toml.{Toml, Value}
import java.nio.file.Paths

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

  private def configPath(mapping: BallIdMapping): Option[String] = {
    mapping.config.map { config =>
      val path = Paths.get(config)
      if (path.isAbsolute) {
        throw new RuntimeException(s"Ball ${mapping.ballName} config must be relative to its balldomain TOML: $config")
      }
      if (mapping.configBaseDir.isEmpty) {
        throw new RuntimeException(s"Ball ${mapping.ballName} config has no balldomain base directory")
      }
      Paths.get(mapping.configBaseDir).resolve(path).normalize().toString
    }
  }

  def ballTable(b: GlobalConfig, ballName: String): Map[String, Value] = {
    val mapping = b.ballDomain.ballIdMappings.find(_.ballName == ballName) match {
      case Some(m) => m
      case None    => throw new RuntimeException(s"No ballIdMapping for ballName=$ballName")
    }
    configPath(mapping) match {
      case Some(path) => ball(load(path))
      case None       => throw new RuntimeException(s"Ball $ballName has no config")
    }
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
