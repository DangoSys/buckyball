package framework.system.configloader

import toml.{Toml, Value}
import framework.system.tile.PrivateDCacheParams
import framework.system.core.configs.RocketCoreParam
import framework.balldomain.configs.{BallDomainParam, BallISAEntry, BallIdMapping}
import framework.top.GlobalConfig
import java.nio.file.{Path, Paths}

/**
 * Unified TOML-based configuration loader for all examples.
 *
 * Supports hierarchical TOML files via `include` directive — sub-tables can
 * be replaced with `include = "relative/path.toml"` to load that file's root
 * table in place. Paths are resolved relative to the directory of the file
 * that contains the include, so file moves stay local.
 *
 * Two convenience shortcuts cut down repetition for homogeneous topologies:
 *   * `[tileTemplate] count = N` — repeat one tile N times
 *   * `[coreTemplate] count = N` — repeat one core N times within a tile
 *
 * Example directory layout:
 * ```
 * examples/goban/
 *   goban_24x16.toml             # [top]+[tileTemplate count=24 include=...]
 *   tiles/tile_16core.toml       # [privateDCache]+[coreTemplate count=16 include=...]
 *   cores/core_default.toml      # [balldomain include=...]+[rocketCore]
 *   balldomains/goban.toml       # ballNum, ballIdMappings, ballISA
 * ```
 */
object TomlConfigLoader {

  /**
   * Load complete example topology from a TOML file (with possible includes).
   */
  def load(tomlPath: String): ExampleTopology = {
    val root    = parseFile(tomlPath)
    val baseDir = Paths.get(tomlPath).toAbsolutePath.getParent

    // Parse top-level
    val nTiles = getInt(root, "top.nTiles")

    // Parse tiles: either [[tiles]] (explicit) or [tileTemplate] count=N (template+repeat)
    val tiles = parseTiles(root, baseDir)

    require(
      tiles.size == nTiles,
      s"TOML declares top.nTiles=$nTiles but defines ${tiles.size} tile(s) in $tomlPath"
    )

    ExampleTopology(tiles)
  }

  // ---------------------------------------------------------------------------
  // Hierarchical parsing
  // ---------------------------------------------------------------------------

  /**
   * Parse a TOML file and return its root table.
   */
  private def parseFile(path: String): Value = {
    val content = scala.io.Source.fromFile(path, "UTF-8").mkString
    Toml.parse(content) match {
      case Right(r)          => r
      case Left((addr, msg)) =>
        throw new RuntimeException(s"TOML parse error at $addr: $msg in $path")
    }
  }

  /**
   * If `table` has an `include` key, load that file (relative to baseDir) and
   * return its root table merged with the rest of `table` (rest wins). Otherwise
   * return `table` unchanged.
   *
   * Returns (resolvedTable, newBaseDir) where newBaseDir is the directory of
   * the included file (so nested includes resolve correctly).
   */
  private def resolveInclude(table: Map[String, Value], baseDir: Path): (Map[String, Value], Path) = {
    table.get("include") match {
      case Some(Value.Str(relPath)) =>
        val includedPath = baseDir.resolve(relPath).normalize()
        val included     = parseFile(includedPath.toString) match {
          case Value.Tbl(t) => t
          case _            => throw new RuntimeException(s"Included file $includedPath must contain a top-level table")
        }
        // Merge: included is the base, local overrides win (except `include` itself)
        val merged       = included ++ (table - "include")
        (merged, includedPath.getParent)
      case None                     => (table, baseDir)
      case _                        => throw new RuntimeException("'include' must be a string")
    }
  }

  private def parseTiles(root: Value, baseDir: Path): Seq[TileTopology] = {
    val rootTable = asTable(root, "root")

    rootTable.get("tileTemplate") match {
      case Some(templateValue) =>
        // tileTemplate + count: repeat one tile N times
        val (templateTable, newBase) = resolveInclude(asTable(templateValue, "tileTemplate"), baseDir)
        val count                    = getInt(templateTable, "count")
        val tile                     = parseTile(templateTable - "count", newBase)
        Seq.fill(count)(tile)

      case None =>
        // Explicit [[tiles]] array
        rootTable.get("tiles") match {
          case Some(Value.Arr(tilesArray)) =>
            tilesArray.map { tileValue =>
              val tileTable           = asTable(tileValue, "tile entry")
              val (resolved, newBase) = resolveInclude(tileTable, baseDir)
              parseTile(resolved, newBase)
            }
          case None                        =>
            throw new RuntimeException("Top-level must define either [[tiles]] array or [tileTemplate]")
          case _                           =>
            throw new RuntimeException("'tiles' must be an array")
        }
    }
  }

  private def parseTile(tileTable: Map[String, Value], baseDir: Path): TileTopology = {
    val privateDCache = getTable(tileTable, "privateDCache").flatMap { dcacheTable =>
      val enable = getBool(dcacheTable, "enable")
      if (!enable) None
      else {
        val ways            = getInt(dcacheTable, "ways")
        val capacityKB      = getInt(dcacheTable, "capacityKB")
        val writeBytes      = getInt(dcacheTable, "writeBytes")
        val portFactor      = getInt(dcacheTable, "portFactor")
        val memCycles       = getInt(dcacheTable, "memCycles")
        val cacheBlockBytes = 64
        val sets            = (capacityKB * 1024) / (cacheBlockBytes * ways)
        Some(PrivateDCacheParams(
          ways = ways,
          sets = sets,
          writeBytes = writeBytes,
          portFactor = portFactor,
          memCycles = memCycles
        ))
      }
    }

    val cores: Seq[Option[GlobalConfig]] = tileTable.get("coreTemplate") match {
      case Some(templateValue) =>
        val (templateTable, newBase) = resolveInclude(asTable(templateValue, "coreTemplate"), baseDir)
        val count                    = getInt(templateTable, "count")
        val coreConfig               = parseCore(templateTable - "count", newBase)
        Seq.fill(count)(coreConfig)

      case None =>
        tileTable.get("cores") match {
          case Some(Value.Arr(coresArray)) =>
            coresArray.map { coreValue =>
              val coreTable           = asTable(coreValue, "core entry")
              val (resolved, newBase) = resolveInclude(coreTable, baseDir)
              parseCore(resolved, newBase)
            }
          case None                        =>
            throw new RuntimeException("Tile must define either [[cores]] array or [coreTemplate]")
          case _                           =>
            throw new RuntimeException("'cores' must be an array")
        }
    }

    // BBTile requires every core's top.nCores to equal the tile's core count.
    // Sync it here so users don't need to repeat the count in every per-core TOML.
    val nCores          = cores.size
    val coresWithNCores = cores.map(_.map(cfg => cfg.copy(top = cfg.top.copy(nCores = nCores))))

    TileTopology(coresWithNCores, privateDCache)
  }

  private def parseCore(coreTable: Map[String, Value], baseDir: Path): Option[GlobalConfig] = {
    // A core entry with `balldomain` becomes a Buckyball-bearing core.
    // Omitting `balldomain` (or empty string) drops the accelerator slot.
    val balldomainOpt = coreTable.get("balldomain").flatMap {
      case Value.Str(s) if s.nonEmpty => Some(parseBallDomain(parseFile(baseDir.resolve(s).toString)))
      case Value.Str(_)               => None // empty string = no balldomain
      case Value.Tbl(t)               =>
        // Inline or include
        val (resolved, _) = resolveInclude(t, baseDir)
        Some(parseBallDomainTable(resolved))
      case _                          => throw new RuntimeException("'balldomain' must be a string path or table")
    }

    balldomainOpt.map { balldomain =>
      val rocketCoreTable = getTable(coreTable, "rocketCore").getOrElse(
        throw new RuntimeException("Core with balldomain must have [rocketCore] section")
      )
      GlobalConfig().copy(
        ballDomain = balldomain,
        rocketCore = parseRocketCore(rocketCoreTable)
      )
    }
  }

  private def parseRocketCore(table: Map[String, Value]): RocketCoreParam = {
    RocketCoreParam(
      xLen = getInt(table, "xLen"),
      pgLevels = getInt(table, "pgLevels"),
      useVM = getBool(table, "useVM"),
      useZba = getBool(table, "useZba"),
      useZbb = getBool(table, "useZbb"),
      useZbs = getBool(table, "useZbs"),
      mulDiv = getTable(table, "mulDiv").map(parseMulDiv).getOrElse(framework.system.core.configs.MulDivParam()),
      fpu = getTable(table, "fpu").map(parseFpu).getOrElse(framework.system.core.configs.FPUParam()),
      dcache = getTable(table, "dcache").map(parseDCache).getOrElse(framework.system.core.configs.DCacheParam()),
      icache = getTable(table, "icache").map(parseICache).getOrElse(framework.system.core.configs.ICacheParam()),
      btb = getTable(table, "btb").map(parseBTB).getOrElse(framework.system.core.configs.BTBParam())
    )
  }

  private def parseMulDiv(table: Map[String, Value]): framework.system.core.configs.MulDivParam =
    framework.system.core.configs.MulDivParam(
      enable = getBool(table, "enable"),
      mulUnroll = getInt(table, "mulUnroll"),
      mulEarlyOut = getBool(table, "mulEarlyOut"),
      divEarlyOut = getBool(table, "divEarlyOut")
    )

  private def parseFpu(table: Map[String, Value]): framework.system.core.configs.FPUParam =
    framework.system.core.configs.FPUParam(
      enable = getBool(table, "enable"),
      minFLen = getInt(table, "minFLen"),
      fLen = getInt(table, "fLen")
    )

  private def parseDCache(table: Map[String, Value]): framework.system.core.configs.DCacheParam =
    framework.system.core.configs.DCacheParam(
      nSets = getInt(table, "nSets"),
      nWays = getInt(table, "nWays"),
      nMSHRs = getInt(table, "nMSHRs")
    )

  private def parseICache(table: Map[String, Value]): framework.system.core.configs.ICacheParam =
    framework.system.core.configs.ICacheParam(
      nSets = getInt(table, "nSets"),
      nWays = getInt(table, "nWays")
    )

  private def parseBTB(table: Map[String, Value]): framework.system.core.configs.BTBParam =
    framework.system.core.configs.BTBParam(
      enable = getBool(table, "enable"),
      nEntries = getInt(table, "nEntries"),
      nRAS = getInt(table, "nRAS")
    )

  /** Parse a BallDomain file (returns BallDomainParam from the root table). */
  private def parseBallDomain(value: Value): BallDomainParam = {
    val table = asTable(value, "balldomain root")
    parseBallDomainTable(table)
  }

  private def parseBallDomainTable(table: Map[String, Value]): BallDomainParam = {
    val ballNum        = getInt(table, "ballNum")
    val ballIdMappings = getArray(table, "ballIdMappings").map { v =>
      val t = asTable(v, "ballIdMapping entry")
      BallIdMapping(
        ballId = getInt(t, "ballId"),
        ballName = getString(t, "ballName"),
        ballClass = getString(t, "ballClass"),
        inBW = getInt(t, "inBW"),
        outBW = getInt(t, "outBW")
      )
    }
    val ballISA        = getArray(table, "ballISA").map { v =>
      val t = asTable(v, "ballISA entry")
      BallISAEntry(
        mnemonic = getString(t, "mnemonic"),
        funct7 = getInt(t, "funct7"),
        bid = getInt(t, "bid")
      )
    }
    BallDomainParam(ballNum = ballNum, ballIdMappings = ballIdMappings, ballISA = ballISA)
  }

  // ---------------------------------------------------------------------------
  // TOML value extraction helpers
  // ---------------------------------------------------------------------------

  private def asTable(value: Value, what: String): Map[String, Value] = value match {
    case Value.Tbl(t) => t
    case _            => throw new RuntimeException(s"Expected $what to be a TOML table")
  }

  private def getTable(table: Map[String, Value], key: String): Option[Map[String, Value]] =
    table.get(key) match {
      case Some(Value.Tbl(t)) => Some(t)
      case None               => None
      case _                  => throw new RuntimeException(s"Expected table at key '$key'")
    }

  private def getArray(table: Map[String, Value], key: String): Seq[Value] =
    table.get(key) match {
      case Some(Value.Arr(arr)) => arr
      case None                 => throw new RuntimeException(s"Missing array at key '$key'")
      case _                    => throw new RuntimeException(s"Expected array at key '$key'")
    }

  private def getString(table: Map[String, Value], key: String): String =
    table.get(key) match {
      case Some(Value.Str(s)) => s
      case None               => throw new RuntimeException(s"Missing string at key '$key'")
      case _                  => throw new RuntimeException(s"Expected string at key '$key'")
    }

  private def getInt(table: Map[String, Value], key: String): Int =
    table.get(key) match {
      case Some(Value.Num(n)) => n.toInt
      case None               => throw new RuntimeException(s"Missing integer at key '$key'")
      case _                  => throw new RuntimeException(s"Expected integer at key '$key'")
    }

  private def getInt(value: Value, path: String): Int = {
    val parts      = path.split('.')
    val finalValue = parts.foldLeft(Option(value)) {
      case (Some(Value.Tbl(t)), key) => t.get(key)
      case _                         => None
    }
    finalValue match {
      case Some(Value.Num(n)) => n.toInt
      case _                  => throw new RuntimeException(s"Missing or invalid integer at path '$path'")
    }
  }

  private def getBool(table: Map[String, Value], key: String): Boolean =
    table.get(key) match {
      case Some(Value.Bool(b)) => b
      case None                => throw new RuntimeException(s"Missing boolean at key '$key'")
      case _                   => throw new RuntimeException(s"Expected boolean at key '$key'")
    }

}

/** Top-level example topology loaded from TOML. */
case class ExampleTopology(tiles: Seq[TileTopology])

/**
 * Per-tile topology: cores + optional privateDCache.
 *
 * @param cores         One entry per core in this tile. `None` disables the
 *                      Buckyball slot for that core (Rocket-only).
 * @param privateDCache Resolved per-tile private DCache parameters, or `None`
 *                      to skip the private DCache layer entirely.
 */
case class TileTopology(
  cores:         Seq[Option[GlobalConfig]],
  privateDCache: Option[PrivateDCacheParams])
