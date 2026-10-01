package memcore.memory.cache.configs

import chisel3.util.{isPow2, log2Ceil}
import scala.io.Source
import scala.util.Using
import toml.{Toml, Value}

case class CacheParams(
  addressBits:   Int,
  lineBytes:     Int,
  sets:          Int,
  ways:          Int,
  idBits:        Int,
  metadataBits:  Int,
  responseDepth: Int) {
  require(lineBytes >= 8 && isPow2(lineBytes))
  require(sets >= 2 && isPow2(sets))
  require(ways >= 1 && isPow2(ways))
  require(idBits > 0 && metadataBits > 0 && responseDepth >= 2)
  val offsetBits = log2Ceil(lineBytes)
  val setBits    = log2Ceil(sets)
  val wayBits    = math.max(1, log2Ceil(ways))
  val tagBits    = addressBits - offsetBits - setBits
  val lineBits   = lineBytes * 8
  require(tagBits > 0)
}

object CacheParams {

  def load(path: String): CacheParams = {
    val text  = Using.resource(Source.fromFile(path, "UTF-8"))(_.mkString)
    val table = Toml.parse(text) match {
      case Right(Value.Tbl(root)) => root("cache") match {
          case Value.Tbl(table) => table
          case value            => throw new IllegalArgumentException(s"Expected [cache] table, got $value")
        }
      case other                  => throw new IllegalArgumentException(s"Invalid cache configuration $path: $other")
    }
    def int(name: String): Int = table(name) match {
      case Value.Num(n) if n == n.toInt => n.toInt
      case value                        => throw new IllegalArgumentException(s"Expected integer cache.$name, got $value")
    }
    CacheParams(
      int("addressBits"),
      int("lineBytes"),
      int("sets"),
      int("ways"),
      int("idBits"),
      int("metadataBits"),
      int("responseDepth")
    )
  }

}
