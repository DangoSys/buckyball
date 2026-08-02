package examples.balls.matrix

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorBackendAnnotation
import framework.balldomain.prototype.systolicarray.SystolicArrayConst
import framework.system.configloader.TomlConfigLoader
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable
import scala.util.Random

class MatrixUnitThroughputSpec extends AnyFlatSpec with ChiselScalatestTester {
  private val tile = 16
  private val bankEntries = 128
  private val wsTaskCount = 10
  private val taskCount = 20

  private case class Shape(m: Int, n: Int, k: Int)
  private case class Task(shape: Shape, mode: Int, bank: Int, aSeed: Long, bSeed: Long)
  private case class BankKey(group: Int, bank: Int, addr: Int)
  private case class ExpectedBeat(
    robId: Int,
    key: BankKey,
    port: Int,
    data: BigInt,
    mask: Vector[Boolean])

  private val defaultWsShapes =
    "129x1x31,145x7x17,161x16x33,177x17x19,193x31x47," +
      "209x4x63,225x8x24,241x15x48,257x32x32,133x33x18"
  private val defaultOsShapes =
    "17x33x19,31x17x32,65x7x17,33x33x33,16x16x16," +
      "129x1x31,7x65x19,48x32x17,19x31x15,64x17x33"

  private def ceilDiv(value: Int, divisor: Int): Int =
    (value + divisor - 1) / divisor

  private def envLong(name: String, default: Long): Long =
    sys.env.get(name).filter(_.nonEmpty).map(_.toLong).getOrElse(default)

  private def envInt(name: String, default: Int): Int =
    sys.env.get(name).filter(_.nonEmpty).map(_.toInt).getOrElse(default)

  private def parseShapes(name: String, default: String): Vector[Shape] = {
    val shapes = sys.env.getOrElse(name, default).split(',').toVector.map { entry =>
      entry.trim.split("[xX]").toVector match {
        case Vector(m, n, k) => Shape(m.toInt, n.toInt, k.toInt)
        case _ => throw new IllegalArgumentException(
          s"$name entry '$entry' must use MxNxK format")
      }
    }
    require(shapes.size == wsTaskCount,
      s"$name must contain exactly $wsTaskCount shapes, found ${shapes.size}")
    shapes.foreach { shape =>
      require(shape.m > 0 && shape.n > 0 && shape.k > 0,
        s"$name dimensions must be positive: $shape")
      require(shape.m < 4096 && shape.n < 4096 && shape.k < 4096,
        s"$name dimensions must fit the 12-bit instruction fields: $shape")
    }
    shapes
  }

  private def signedMatrix(rows: Int, cols: Int, seed: Long): Vector[Int] = {
    val random = new Random(seed)
    Vector.fill(rows * cols) {
      val magnitude = random.nextInt(63) + 1
      if (random.nextBoolean()) magnitude else -magnitude
    }
  }

  private def golden(a: Vector[Int], b: Vector[Int], shape: Shape): Vector[Int] =
    Vector.tabulate(shape.m * shape.n) { index =>
      val row = index / shape.n
      val col = index % shape.n
      (0 until shape.k).foldLeft(0L) { (sum, inner) =>
        sum + a(row * shape.k + inner).toLong * b(inner * shape.n + col).toLong
      }.toInt
    }

  private def packI8(values: Seq[Int]): BigInt =
    values.zipWithIndex.foldLeft(BigInt(0)) { case (word, (value, lane)) =>
      word | (BigInt(value & 0xff) << (lane * 8))
    }

  private def packI32(values: Seq[Int]): BigInt =
    values.zipWithIndex.foldLeft(BigInt(0)) { case (word, (value, lane)) =>
      word | (BigInt(value.toLong & 0xffffffffL) << (lane * 32))
    }

  private def operandKey(bank: Int, linearAddress: Int): BankKey =
    BankKey(linearAddress / bankEntries, bank, linearAddress % bankEntries)

  private final class VirtualMemory(bankNum: Int) {
    private val words = mutable.Map.empty[BankKey, BigInt]

    private def validate(key: BankKey): Unit = {
      require(key.group >= 0 && key.group < bankNum, s"invalid group in $key")
      require(key.bank >= 0 && key.bank < bankNum, s"invalid bank in $key")
      require(key.addr >= 0 && key.addr < bankEntries, s"invalid address in $key")
    }

    def read(key: BankKey): BigInt = {
      validate(key)
      words.getOrElse(key, BigInt(0))
    }

    def write(key: BankKey, data: BigInt): Unit = {
      validate(key)
      words(key) = data
    }

    def writeMasked(key: BankKey, data: BigInt, mask: Seq[Boolean]): Unit = {
      val merged = mask.zipWithIndex.foldLeft(read(key)) { case (word, (enabled, byte)) =>
        if (!enabled) word
        else {
          val byteMask = BigInt(0xff) << (byte * 8)
          (word & ~byteMask) | (data & byteMask)
        }
      }
      write(key, merged)
    }
  }

  private def loadOperands(
    aMemory: VirtualMemory,
    bMemory: VirtualMemory,
    task: Task,
    a: Vector[Int],
    b: Vector[Int]
  ): Unit = {
    val shape = task.shape
    val mTiles = ceilDiv(shape.m, tile)
    val nTiles = ceilDiv(shape.n, tile)
    val kTiles = ceilDiv(shape.k, tile)

    for (mt <- 0 until mTiles; kt <- 0 until kTiles) {
      val validRows = math.min(tile, shape.m - mt * tile)
      val tileBase = (mt * kTiles + kt) * tile
      for (row <- 0 until validRows) {
        val values = Vector.tabulate(tile) { lane =>
          val sourceRow = mt * tile + row
          val sourceCol = kt * tile + lane
          if (sourceCol < shape.k) a(sourceRow * shape.k + sourceCol) else 0
        }
        aMemory.write(operandKey(task.bank, tileBase + row), packI8(values))
      }
    }

    for (nt <- 0 until nTiles; kt <- 0 until kTiles) {
      val validRows = math.min(tile, shape.k - kt * tile)
      val tileBase = (nt * kTiles + kt) * tile
      for (row <- 0 until validRows) {
        val values = Vector.tabulate(tile) { lane =>
          val sourceRow = kt * tile + row
          val sourceCol = nt * tile + lane
          if (sourceCol < shape.n) b(sourceRow * shape.n + sourceCol) else 0
        }
        bMemory.write(operandKey(task.bank, tileBase + row), packI8(values))
      }
    }
  }

  private def expectedBeats(task: Task, robId: Int, c: Vector[Int]): Vector[ExpectedBeat] = {
    val shape = task.shape
    val nTiles = ceilDiv(shape.n, tile)
    val beats = mutable.ArrayBuffer.empty[ExpectedBeat]

    val mTiles = ceilDiv(shape.m, tile)
    val tileOrder = if (task.mode == 1) {
      for (nt <- 0 until nTiles; mt <- 0 until mTiles) yield (mt, nt)
    } else {
      for (mt <- 0 until mTiles; nt <- 0 until nTiles) yield (mt, nt)
    }

    for ((mt, nt) <- tileOrder) {
      val validRows = math.min(tile, shape.m - mt * tile)
      val validCols = math.min(tile, shape.n - nt * tile)
      for (row <- 0 until validRows; port <- 0 until ceilDiv(validCols, 4)) {
        val validElems = math.min(4, validCols - port * 4)
        val values = Vector.tabulate(validElems) { lane =>
          val matrixRow = mt * tile + row
          val matrixCol = nt * tile + port * 4 + lane
          c(matrixRow * shape.n + matrixCol)
        }
        val rowAddress = mt * tile * nTiles + nt * validRows + row
        val groupBase = (rowAddress / bankEntries) * SystolicArrayConst.StoreWritePorts
        beats += ExpectedBeat(
          robId,
          BankKey(groupBase + port, task.bank, rowAddress % bankEntries),
          port,
          packI32(values),
          Vector.tabulate(16)(_ < validElems * 4))
      }
    }
    beats.toVector
  }

  private def idealARows(task: Task): Long =
    task.shape.m.toLong * ceilDiv(task.shape.n, tile) * ceilDiv(task.shape.k, tile)

  private def commandWords(task: Task): (BigInt, BigInt) = {
    val rs1 = BigInt(task.bank) | (BigInt(task.bank) << 10) | (BigInt(task.bank) << 20)
    val shape = task.shape
    val rs2 = BigInt(shape.m) | (BigInt(shape.n) << 12) | (BigInt(shape.k) << 24) |
      (BigInt(task.mode) << 36)
    (rs1, rs2)
  }

  behavior of "MatrixUnit"

  it should "run ordered WS and OS throughput batches against virtual-bank goldens" in {
    val seed = envLong("MATRIX_TEST_SEED", 97L)
    val wsShapes = parseShapes("MATRIX_WS_SHAPES", defaultWsShapes)
    val osShapes = parseShapes("MATRIX_OS_SHAPES", defaultOsShapes)
    val dataSeeds = new Random(seed)
    val tasks = (wsShapes.map(_ -> 1) ++ osShapes.map(_ -> 0)).zipWithIndex.map {
      case ((shape, mode), index) =>
        Task(shape, mode, index, dataSeeds.nextLong(), dataSeeds.nextLong())
    }

    val configPath = "../examples/chips/toy/configs/toy.toml"
    val baseConfig = TomlConfigLoader.load(configPath).tiles.head.cores.head.get
    val config = baseConfig.copy(frontend = baseConfig.frontend.copy(rob_entries = 32))
    require(config.memDomain.bankEntries == bankEntries,
      s"test expects $bankEntries bank entries")
    require(config.memDomain.bankNum >= taskCount,
      s"test requires at least $taskCount virtual banks")
    tasks.foreach { task =>
      val shape = task.shape
      val operandCapacity = config.memDomain.bankNum * bankEntries
      val resultCapacity = (config.memDomain.bankNum /
        SystolicArrayConst.StoreWritePorts) * bankEntries
      require(ceilDiv(shape.m, tile) * ceilDiv(shape.k, tile) * tile <= operandCapacity,
        s"A shape exceeds virtual-bank address capacity: $shape")
      require(ceilDiv(shape.n, tile) * ceilDiv(shape.k, tile) * tile <= operandCapacity,
        s"B shape exceeds virtual-bank address capacity: $shape")
      require(shape.m * ceilDiv(shape.n, tile) <= resultCapacity,
        s"C shape exceeds virtual-bank address capacity: $shape")
    }

    val aMemory = new VirtualMemory(config.memDomain.bankNum)
    val bMemory = new VirtualMemory(config.memDomain.bankNum)
    val cMemory = new VirtualMemory(config.memDomain.bankNum)
    val expectedByRob = tasks.zipWithIndex.map { case (task, robId) =>
      val a = signedMatrix(task.shape.m, task.shape.k, task.aSeed)
      val b = signedMatrix(task.shape.k, task.shape.n, task.bSeed)
      loadOperands(aMemory, bMemory, task, a, b)
      expectedBeats(task, robId, golden(a, b, task.shape))
    }
    val expectedWriteOrder = expectedByRob.flatten
    expectedWriteOrder.foreach(beat => cMemory.write(beat.key, BigInt(0)))

    val maxCycles = envInt("MATRIX_MAX_CYCLES", 200000)
    test(new MatrixUnit(config)).withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
      dut.clock.setTimeout(0)
      val readResponses = Vector.fill(2)(mutable.Queue.empty[BigInt])
      val writeResponses = Vector.fill(4)(mutable.Queue.empty[Boolean])
      val commandCycles = Array.fill[Option[Int]](taskCount)(None)
      val completionCycles = Array.fill[Option[Int]](taskCount)(None)
      var nextCommand = 0
      var nextCompletion = 0
      var nextWrite = 0
      var cycle = 0

      dut.reset.poke(true.B)
      dut.io.cmdReq.valid.poke(false.B)
      dut.io.cmdResp.ready.poke(true.B)
      for (port <- 0 until 2) {
        dut.io.bankRead(port).io.req.ready.poke(true.B)
        dut.io.bankRead(port).io.resp.valid.poke(false.B)
        dut.io.bankRead(port).io.resp.bits.data.poke(0.U)
      }
      for (port <- 0 until 4) {
        dut.io.bankWrite(port).io.req.ready.poke(true.B)
        dut.io.bankWrite(port).io.resp.valid.poke(false.B)
        dut.io.bankWrite(port).io.resp.bits.ok.poke(true.B)
      }
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step()

      while (nextCompletion < taskCount && cycle < maxCycles) {
        val mayIssue = nextCommand < taskCount &&
          (nextCommand < wsTaskCount || nextCompletion >= wsTaskCount)
        if (mayIssue) {
          val (rs1, rs2) = commandWords(tasks(nextCommand))
          dut.io.cmdReq.valid.poke(true.B)
          dut.io.cmdReq.bits.cmd.bid.poke(0.U)
          dut.io.cmdReq.bits.cmd.funct7.poke(65.U)
          dut.io.cmdReq.bits.cmd.iter.poke(0.U)
          dut.io.cmdReq.bits.cmd.op1_en.poke(false.B)
          dut.io.cmdReq.bits.cmd.op2_en.poke(false.B)
          dut.io.cmdReq.bits.cmd.wr_spad_en.poke(false.B)
          dut.io.cmdReq.bits.cmd.op1_from_spad.poke(false.B)
          dut.io.cmdReq.bits.cmd.op2_from_spad.poke(false.B)
          dut.io.cmdReq.bits.cmd.special.poke(0.U)
          dut.io.cmdReq.bits.cmd.op1_bank.poke(0.U)
          dut.io.cmdReq.bits.cmd.op2_bank.poke(0.U)
          dut.io.cmdReq.bits.cmd.wr_bank.poke(0.U)
          dut.io.cmdReq.bits.cmd.op1_col.poke(0.U)
          dut.io.cmdReq.bits.cmd.op2_col.poke(0.U)
          dut.io.cmdReq.bits.cmd.wr_col.poke(0.U)
          dut.io.cmdReq.bits.cmd.meta_bank.poke(0.U)
          dut.io.cmdReq.bits.cmd.rs1.poke(rs1.U)
          dut.io.cmdReq.bits.cmd.rs2.poke(rs2.U)
          dut.io.cmdReq.bits.rob_id.poke(nextCommand.U)
          dut.io.cmdReq.bits.is_sub.poke(false.B)
          dut.io.cmdReq.bits.sub_rob_id.poke(0.U)
        } else {
          dut.io.cmdReq.valid.poke(false.B)
        }

        for (port <- 0 until 2) {
          dut.io.bankRead(port).io.req.ready.poke(true.B)
          dut.io.bankRead(port).io.resp.valid.poke(readResponses(port).nonEmpty.B)
          dut.io.bankRead(port).io.resp.bits.data.poke(
            readResponses(port).headOption.getOrElse(BigInt(0)).U)
        }
        for (port <- 0 until 4) {
          dut.io.bankWrite(port).io.req.ready.poke(true.B)
          dut.io.bankWrite(port).io.resp.valid.poke(writeResponses(port).nonEmpty.B)
          dut.io.bankWrite(port).io.resp.bits.ok.poke(true.B)
        }

        val commandFire = dut.io.cmdReq.valid.peekBoolean() && dut.io.cmdReq.ready.peekBoolean()
        val completionFire = dut.io.cmdResp.valid.peekBoolean() && dut.io.cmdResp.ready.peekBoolean()
        val readRequestFire = Vector.tabulate(2)(port =>
          dut.io.bankRead(port).io.req.valid.peekBoolean() &&
            dut.io.bankRead(port).io.req.ready.peekBoolean())
        val readResponseFire = Vector.tabulate(2)(port =>
          dut.io.bankRead(port).io.resp.valid.peekBoolean() &&
            dut.io.bankRead(port).io.resp.ready.peekBoolean())
        val writeRequestFire = Vector.tabulate(4)(port =>
          dut.io.bankWrite(port).io.req.valid.peekBoolean() &&
            dut.io.bankWrite(port).io.req.ready.peekBoolean())
        val writeResponseFire = Vector.tabulate(4)(port =>
          dut.io.bankWrite(port).io.resp.valid.peekBoolean() &&
            dut.io.bankWrite(port).io.resp.ready.peekBoolean())

        if (completionFire) {
          val actualRob = dut.io.cmdResp.bits.rob_id.peekInt().toInt
          assert(actualRob == nextCompletion,
            s"completion order mismatch at cycle $cycle: actual ROB $actualRob, expected $nextCompletion")
          completionCycles(nextCompletion) = Some(cycle)
        }

        val readWords = Vector.tabulate(2) { port =>
          if (readRequestFire(port)) {
            val key = BankKey(
              dut.io.bankRead(port).group_id.peekInt().toInt,
              dut.io.bankRead(port).bank_id.peekInt().toInt,
              dut.io.bankRead(port).io.req.bits.addr.peekInt().toInt)
            Some(if (port == 0) aMemory.read(key) else bMemory.read(key))
          } else None
        }

        val writes = Vector.tabulate(4) { port =>
          if (writeRequestFire(port)) {
            assert(nextWrite < expectedWriteOrder.size,
              s"unexpected extra C write at cycle $cycle on port $port")
            val actualKey = BankKey(
              dut.io.bankWrite(port).group_id.peekInt().toInt,
              dut.io.bankWrite(port).bank_id.peekInt().toInt,
              dut.io.bankWrite(port).io.req.bits.addr.peekInt().toInt)
            val actualRob = dut.io.bankWrite(port).rob_id.peekInt().toInt
            val actualData = dut.io.bankWrite(port).io.req.bits.data.peekInt()
            val actualMask = Vector.tabulate(16)(index =>
              dut.io.bankWrite(port).io.req.bits.mask(index).peekBoolean())
            val expected = expectedWriteOrder(nextWrite)
            assert(actualRob == expected.robId && actualKey == expected.key && port == expected.port,
              s"C write order mismatch at sequence $nextWrite cycle $cycle: " +
                s"actual=(ROB $actualRob,$actualKey,port $port), " +
                s"expected=(ROB ${expected.robId},${expected.key},port ${expected.port})")
            assert(actualMask == expected.mask,
              s"C write mask mismatch at sequence $nextWrite: actual=$actualMask expected=${expected.mask}")
            for (byte <- actualMask.indices if actualMask(byte)) {
              val actualByte = (actualData >> (byte * 8)) & BigInt(0xff)
              val expectedByte = (expected.data >> (byte * 8)) & BigInt(0xff)
              assert(actualByte == expectedByte,
                s"C write data mismatch at sequence $nextWrite byte $byte: " +
                  s"actual=$actualByte expected=$expectedByte")
            }
            nextWrite += 1
            Some((actualKey, actualData, actualMask))
          } else None
        }

        dut.clock.step()

        if (commandFire) {
          commandCycles(nextCommand) = Some(cycle)
          nextCommand += 1
        }
        if (completionFire) nextCompletion += 1
        for (port <- 0 until 2) {
          if (readResponseFire(port)) readResponses(port).dequeue()
          readWords(port).foreach(readResponses(port).enqueue(_))
        }
        for (port <- 0 until 4) {
          if (writeResponseFire(port)) writeResponses(port).dequeue()
          writes(port).foreach { case (key, data, mask) =>
            cMemory.writeMasked(key, data, mask)
            writeResponses(port).enqueue(true)
          }
        }
        cycle += 1
      }

      assert(nextCompletion == taskCount,
        s"timeout after $cycle cycles: commands=$nextCommand/$taskCount, " +
          s"completions=$nextCompletion/$taskCount, writes=$nextWrite/${expectedWriteOrder.size}")
      assert(nextWrite == expectedWriteOrder.size,
        s"missing C writes: observed=$nextWrite expected=${expectedWriteOrder.size}")

      expectedWriteOrder.foreach { expected =>
        val actual = cMemory.read(expected.key)
        for (byte <- expected.mask.indices if expected.mask(byte)) {
          val actualByte = (actual >> (byte * 8)) & BigInt(0xff)
          val expectedByte = (expected.data >> (byte * 8)) & BigInt(0xff)
          assert(actualByte == expectedByte,
            s"final C mismatch at ${expected.key} byte $byte: actual=$actualByte expected=$expectedByte")
        }
      }

      def cycleOf(values: Array[Option[Int]], index: Int, label: String): Int =
        values(index).getOrElse(throw new AssertionError(s"missing $label cycle for ROB $index"))

      val wsStart = cycleOf(commandCycles, 0, "command")
      val wsDone = cycleOf(completionCycles, wsTaskCount - 1, "completion")
      val osStart = cycleOf(commandCycles, wsTaskCount, "command")
      val osDone = cycleOf(completionCycles, taskCount - 1, "completion")
      assert(osStart > wsDone,
        s"OS batch started at cycle $osStart before WS completed at cycle $wsDone")
      val wsCycles = wsDone - wsStart + 1L
      val osCycles = osDone - osStart + 1L
      val wsRows = tasks.take(wsTaskCount).map(idealARows).sum
      val osRows = tasks.drop(wsTaskCount).map(idealARows).sum

      println(s"PASS seed=$seed")
      println(f"WS cycles=$wsCycles A_rows=$wsRows throughput=${wsRows.toDouble / wsCycles}%.4f")
      println(f"OS cycles=$osCycles A_rows=$osRows throughput=${osRows.toDouble / osCycles}%.4f")
    }
  }
}
