package memcore.memory.coherence.configs

import chisel3.util.log2Ceil
import java.nio.file.Paths
import scala.io.Source
import scala.util.Using
import toml.{Toml, Value}
import memcore.bus.chi.Params
import memcore.memory.cache.configs.CacheParams

case class CoherenceParams(
  chi:         Params,
  cache:       CacheParams,
  agents:      Int,
  mshrEntries: Int,
  homeId:      Int) {
  require(cache.lineBytes == 64 && cache.addressBits == chi.addressBits)
  require(agents >= 1 && agents < homeId && BigInt(homeId) < (BigInt(1) << chi.nodeIdBits))
  require(mshrEntries >= 1 && BigInt(mshrEntries) <= (BigInt(1) << chi.txnIdBits))
  require(BigInt(mshrEntries) <= (BigInt(1) << cache.idBits))
  val slotBits        = math.max(1, log2Ceil(mshrEntries))
  val lineAddressBits = chi.addressBits - cache.offsetBits
}

object CoherenceParams {

  def load(path: String): CoherenceParams = {
    val text      = Using.resource(Source.fromFile(path, "UTF-8"))(_.mkString)
    val table     = Toml.parse(text) match {
      case Right(Value.Tbl(root)) => root("coherence") match {
          case Value.Tbl(t) => t
          case value        => throw new IllegalArgumentException(s"Expected [coherence], got $value")
        }
      case value                  => throw new IllegalArgumentException(s"Invalid coherence config: $value")
    }
    def int(name: String): Int = table(name) match {
      case Value.Num(n) if n == n.toInt => n.toInt
      case value                        => throw new IllegalArgumentException(s"Expected integer $name, got $value")
    }
    val cachePath = table("cacheConfig") match {
      case Value.Str(value) => Paths.get(path).toAbsolutePath.getParent.resolve(value).normalize.toString
      case value            => throw new IllegalArgumentException(s"Expected cacheConfig path, got $value")
    }
    CoherenceParams(
      Params(int("nodeIdBits"), int("addressBits"), int("dataBits")),
      CacheParams.load(cachePath),
      int("agents"),
      int("mshrEntries"),
      int("homeId")
    )
  }

}
