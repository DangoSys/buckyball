package examples.balls.mxmm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig
import examples.balls.mxmm.configs.MxmmBallParam

@instantiable
class Unit(b: GlobalConfig) extends Module {
  private val p            = MxmmBallParam(b)
  private val entries      = b.memDomain.bankEntries
  private val addressWidth = log2Ceil(entries)
  require(b.memDomain.bankWidth == 128 && b.memDomain.bankMaskLen == 16)
  require(entries >= 64 && entries % 32 == 0)
  private val mapping = b.ballDomain.ballIdMappings.find(_.ballName == "MxmmBall").get
  require(mapping.inBW == 2 && mapping.outBW == 1)

  @public val io = IO(new Bundle {
    val cmdReq       = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp      = Decoupled(new BallRsComplete(b))
    val bankRead     = Vec(2, Flipped(new BankRead(b)))
    val bankWrite    = Vec(1, Flipped(new BankWrite(b)))
    val channelReady = Input(Bool())
    val status       = new BallStatus
  })

  private val Seq(
    idle,
    waitChannels,
    loadInputs,
    initRequest,
    initResponse,
    compute,
    drain,
    store,
    outputRequest,
    outputResponse,
    outputWrite,
    outputAck,
    complete
  ) = Enum(13)

  private val state       = RegInit(idle)
  private val command     = Reg(new BallRsIssue(b))
  private val m           = Reg(UInt(12.W))
  private val n           = Reg(UInt(12.W))
  private val k           = Reg(UInt(34.W))
  private val mx          = Reg(Bool())
  private val windowed    = Reg(Bool())
  private val separate    = Reg(Bool())
  private val first       = Reg(Bool())
  private val last        = Reg(Bool())
  private val base        = Reg(UInt(6.W))
  private val chain       = RegInit(false.B)
  private val chainM      = Reg(UInt(12.W))
  private val chainN      = Reg(UInt(12.W))
  private val chainFormat = Reg(UInt(7.W))
  private val chainBank   = Reg(UInt(b.memDomain.vbankIdWidth.W))
  private val chainBase   = Reg(UInt(6.W))

  private val nGroups       = n >> 4
  private val panelHeight   = Mux(m === 1.U, 1.U, 16.U)
  private val totalPanels   = Mux(m === 1.U, 1.U, m >> 4) * nGroups
  private val batch         = Reg(UInt(addressWidth.W))
  private val count         = Mux(totalPanels - batch > p.contexts.U, p.contexts.U, totalPanels - batch)
  private val minimumPeriod = Mux(separate, (2 * p.arithmeticLatency + 1).U, (p.arithmeticLatency + 1).U)
  private val period        = Mux(count > minimumPeriod, count, minimumPeriod)
  private val context       = Reg(UInt(log2Ceil(p.contexts).W))
  private val row           = Reg(UInt(4.W))
  private val reduction     = Reg(UInt(34.W))
  private val completed     = Reg(UInt(48.W))
  private val panel         = batch + context
  private val rowGroup      = panel / nGroups
  private val columnGroup   = panel % nGroups
  private val accAddress    = rowGroup * 16.U * nGroups + row * nGroups + columnGroup
  private val accumulator   = SyncReadMem(entries / 4, UInt(512.W))
  private val savedRow      = Reg(UInt(512.W))
  private val outputLine    = Reg(UInt(addressWidth.W))
  private val outputQuarter = Reg(UInt(2.W))

  private val panels    = Instantiate(new Panels(entries))
  private val array     = Instantiate(new Array(p))
  private val readValid = state === compute && context < count
  panels.io.read        := readValid
  panels.io.rowGroup    := rowGroup
  panels.io.columnGroup := columnGroup
  panels.io.k           := reduction
  panels.io.reduction   := k
  panels.io.mxfp8       := mx
  panels.io.vector      := m === 1.U
  array.io.valid        := panels.io.valid
  array.io.a            := panels.io.a
  array.io.b            := panels.io.b
  array.io.separate     := separate
  array.io.vector       := m === 1.U
  val computeContext = RegNext(context)
  array.io.context := Mux(state === compute || state === drain, computeContext, context)
  array.io.row     := row
  array.io.load    := false.B
  array.io.rowData := 0.U

  val accRead = accumulator.read(
    Mux(state === outputRequest, outputLine, accAddress),
    state === outputRequest || (state === initRequest && !first && row < panelHeight)
  )

  private val inputState = Reg(Vec(2, UInt(2.W)))
  private val inputWord  = Reg(Vec(2, UInt(128.W)))
  private val inputIndex = Reg(Vec(2, UInt(addressWidth.W)))
  private val inputRow   = Reg(Vec(2, UInt(12.W)))
  private val rowOffset  = Reg(Vec(2, UInt(addressWidth.W)))
  private val scaleMode  = Reg(Vec(2, Bool()))
  private val scaleByte  = Reg(Vec(2, UInt(4.W)))
  private val scaleCount = Reg(Vec(2, UInt((addressWidth + 4).W)))
  private val rowWords   = Mux(mx, k >> 4, k >> 2)
  private val window     = Module(new InputWindow(b))
  window.io.start          := io.cmdReq.fire && io.cmdReq.bits.cmd.funct7 === 75.U
  window.io.enable         := state === loadInputs && windowed
  window.io.rows           := io.cmdReq.bits.cmd.rs2(11, 0)
  window.io.count          := io.cmdReq.bits.cmd.rs1(63, 30)
  window.io.fullK          := io.cmdReq.bits.cmd.rs2(47, 32)
  window.io.startK         := io.cmdReq.bits.cmd.rs2(63, 48)
  window.io.request.ready  := io.bankRead(0).io.req.ready && windowed
  window.io.response.valid := io.bankRead(0).io.resp.valid && windowed
  window.io.response.bits  := io.bankRead(0).io.resp.bits

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := command.rob_id
  io.cmdResp.bits.is_sub     := command.is_sub
  io.cmdResp.bits.sub_rob_id := command.sub_rob_id
  io.status.idle             := state === idle
  io.status.running          := state =/= idle && state =/= complete

  for (operand <- 0 until 2) {
    val rows      = if (operand == 0) m else n
    val codeWords = rows * rowWords
    val scales    = rows * (k >> 5)
    val port      = io.bankRead(operand)
    port.rob_id                   := command.rob_id
    port.ball_id                  := 0.U
    port.group_id                 := 0.U
    port.bank_id                  := (if (operand == 0) command.cmd.op1_bank else command.cmd.op2_bank)
    port.io.req.valid             := state === loadInputs && inputState(operand) === 0.U
    port.io.req.bits.addr         := inputIndex(operand)
    port.io.resp.ready            := state === loadInputs && inputState(operand) === 1.U
    panels.io.write(operand)      := port.io.resp.fire && !scaleMode(operand)
    panels.io.scaleWrite(operand) := state === loadInputs && inputState(operand) === 2.U
    panels.io.lane(operand)       := inputRow(operand)(3, 0)
    panels.io.address(operand)    := (inputRow(operand) >> 4) * Mux(scaleMode(operand), k >> 5, rowWords) + rowOffset(
      operand
    )
    panels.io.word(operand)       := port.io.resp.bits.data
    panels.io.scale(operand)      := (inputWord(operand) >> (scaleByte(operand) << 3))(7, 0)
    val legacy = if (operand == 0) !windowed else true.B
    when(port.io.req.fire && legacy)(inputState(operand) := 1.U)
    when(port.io.resp.fire && legacy) {
      when(scaleMode(operand)) {
        inputWord(operand)  := port.io.resp.bits.data
        scaleByte(operand)  := 0.U
        inputState(operand) := 2.U
      }.otherwise {
        inputIndex(operand)             := inputIndex(operand) + 1.U
        when(rowOffset(operand) + 1.U === rowWords) {
          rowOffset(operand) := 0.U
          inputRow(operand)  := inputRow(operand) + 1.U
        }.otherwise(rowOffset(operand)  := rowOffset(operand) + 1.U)
        when(inputIndex(operand) +& 1.U === codeWords) {
          when(mx) {
            scaleMode(operand)  := true.B
            inputRow(operand)   := 0.U
            rowOffset(operand)  := 0.U
            inputState(operand) := 0.U
          }.otherwise(inputState(operand) := 3.U)
        }.otherwise(inputState(operand) := 0.U)
      }
    }
    when(state === loadInputs && inputState(operand) === 2.U && legacy) {
      scaleCount(operand)            := scaleCount(operand) + 1.U
      scaleByte(operand)             := scaleByte(operand) + 1.U
      when(rowOffset(operand) + 1.U === (k >> 5)) {
        rowOffset(operand) := 0.U
        inputRow(operand)  := inputRow(operand) + 1.U
      }.otherwise(rowOffset(operand) := rowOffset(operand) + 1.U)
      when(scaleCount(operand) + 1.U === scales) {
        inputState(operand) := 3.U
      }.elsewhen(scaleByte(operand) === 15.U) {
        inputIndex(operand) := inputIndex(operand) + 1.U
        inputState(operand) := 0.U
      }
    }
  }

  when(windowed) {
    io.bankRead(0).io.req.valid     := window.io.request.valid
    io.bankRead(0).io.req.bits.addr := window.io.request.bits.addr
    io.bankRead(0).io.resp.ready    := window.io.response.ready
    panels.io.write(0)              := window.io.write
    panels.io.scaleWrite(0)         := window.io.scaleWrite
    panels.io.lane(0)               := window.io.lane
    panels.io.address(0)            := window.io.address
    panels.io.word(0)               := window.io.word
    panels.io.scale(0)              := window.io.scale
  }

  private val writePort = io.bankWrite(0)
  writePort.rob_id           := command.rob_id
  writePort.ball_id          := 0.U
  writePort.group_id         := 0.U
  writePort.bank_id          := command.cmd.wr_bank
  writePort.io.req.valid     := state === outputWrite
  writePort.io.req.bits.addr := base + (outputLine << 2) + outputQuarter
  writePort.io.req.bits.data := (savedRow >> (outputQuarter << 7))(127, 0)
  writePort.io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  writePort.io.resp.ready    := state === outputAck

  when(io.cmdReq.fire) {
    val cmd       = io.cmdReq.bits.cmd
    val rows      = cmd.rs2(11, 0)
    val cols      = cmd.rs2(23, 12)
    val inputK    = cmd.rs1(63, 30)
    val isWindow  = cmd.funct7 === 75.U
    val isMx      = cmd.funct7 === 71.U || isWindow
    assert(isMx || cmd.funct7 === 72.U || cmd.funct7 === 73.U, "MATMUL format is invalid")
    assert(rows === 1.U || (rows =/= 0.U && rows(3, 0) === 0.U), "MATMUL M must be one or a positive multiple of 16")
    assert(cols =/= 0.U && cols(3, 0) === 0.U, "MATMUL N must be a positive multiple of 16")
    assert(inputK =/= 0.U && Mux(isMx, inputK(4, 0) === 0.U, inputK(1, 0) === 0.U), "MATMUL K alignment is invalid")
    when(isWindow) {
      val full  = cmd.rs2(47, 32)
      val start = cmd.rs2(63, 48)
      assert(full =/= 0.U && full(4, 0) === 0.U && start(4, 0) === 0.U, "MATMUL A window alignment is invalid")
      assert(start +& inputK <= full, "MATMUL A window exceeds fullK")
    }.otherwise(assert(cmd.rs2(63, 32) === 0.U, "MATMUL reserves rs2[63:32]"))
    assert(
      cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
      "MATMUL requires distinct banks"
    )
    assert(
      cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 1.U,
      "MATMUL requires one physical bank per operand"
    )
    assert(cmd.rs1(9, 0) < b.memDomain.virtualBankCount.U && cmd.rs1(19, 10) < b.memDomain.virtualBankCount.U && cmd.rs1(
      29,
      20
    ) < b.memDomain.virtualBankCount.U)
    val aElements = rows * Mux(isWindow, cmd.rs2(47, 32), inputK)
    val bElements = cols * inputK
    assert(Mux(isMx, aElements + (aElements >> 5), aElements << 2) <= (entries * 16).U, "MATMUL A exceeds bank capacity")
    assert(Mux(isMx, bElements + (bElements >> 5), bElements << 2) <= (entries * 16).U, "MATMUL B exceeds bank capacity")
    assert((cmd.rs2(31, 26) << 4) +& ((rows * cols) << 2) <= (entries * 16).U, "MATMUL C exceeds bank capacity")
    when(cmd.rs2(24)) {
      assert(!chain, "MATMUL first cannot replace a live chain")
      chain       := true.B
      chainM      := rows
      chainN      := cols
      chainFormat := cmd.funct7
      chainBank   := cmd.wr_bank
      chainBase   := cmd.rs2(31, 26)
    }.otherwise {
      assert(
        chain && chainM === rows && chainN === cols && chainFormat === cmd.funct7 && chainBank === cmd.wr_bank && chainBase === cmd.rs2(
          31,
          26
        ),
        "MATMUL continuation changed its chain"
      )
    }
    command := io.cmdReq.bits
    m         := rows
    n         := cols
    k         := inputK
    mx        := isMx
    windowed  := isWindow
    separate  := cmd.funct7 === 73.U
    first     := cmd.rs2(24)
    last      := cmd.rs2(25)
    base      := cmd.rs2(31, 26)
    batch     := 0.U
    context   := 0.U
    row       := 0.U
    reduction := 0.U
    completed := 0.U
    for (operand <- 0 until 2) {
      inputState(operand) := 0.U
      inputIndex(operand) := 0.U
      inputRow(operand)   := 0.U
      rowOffset(operand)  := 0.U
      scaleMode(operand)  := false.B
      scaleCount(operand) := 0.U
    }
    state := waitChannels
  }

  switch(state) {
    is(waitChannels)(when(io.channelReady)(state := loadInputs))
    is(loadInputs)(
      when(Mux(windowed, window.io.done, inputState(0) === 3.U) && inputState(1) === 3.U)(state := initRequest)
    )
    is(initRequest) {
      when(first || row >= panelHeight) {
        array.io.load   := true.B
        when(row === 15.U) {
          row                  := 0.U
          when(context + 1.U === count) { context := 0.U; state := compute }
            .otherwise(context := context + 1.U)
        }.otherwise(row := row + 1.U)
      }.otherwise(state := initResponse)
    }
    is(initResponse) {
      array.io.load    := true.B
      array.io.rowData := accRead
      when(row === 15.U) {
        row := 0.U
        when(context + 1.U === count) { context := 0.U; state := compute }
          .otherwise { context := context + 1.U; state := initRequest }
      }.otherwise { row := row + 1.U; state := initRequest }
    }
    is(compute) {
      when(context + 1.U === period) {
        context                := 0.U
        when(reduction + 1.U === k)(state := drain)
          .otherwise(reduction := reduction + 1.U)
      }.otherwise(context := context + 1.U)
    }
    is(store) {
      when(row < panelHeight)(accumulator.write(accAddress, array.io.rowOut))
      when(row === 15.U) {
        row                 := 0.U
        when(context + 1.U === count) {
          context := 0.U
          when(batch + count === totalPanels) {
            when(last) { outputLine := 0.U; outputQuarter := 0.U; state := outputRequest }
              .otherwise(state := complete)
          }.otherwise {
            batch     := batch + count
            reduction := 0.U
            completed := 0.U
            state     := initRequest
          }
        }.otherwise(context := context + 1.U)
      }.otherwise(row := row + 1.U)
    }
    is(outputRequest)(state := outputResponse)
    is(outputResponse) { savedRow := accRead; state := outputWrite }
    is(outputWrite)(when(writePort.io.req.fire)(state := outputAck))
    is(outputAck) {
      when(writePort.io.resp.fire) {
        assert(writePort.io.resp.bits.ok, "MATMUL bank write failed")
        when(outputQuarter === 3.U) {
          outputQuarter := 0.U
          when(outputLine + 1.U === m * nGroups) { chain := false.B; state := complete }
            .otherwise { outputLine := outputLine + 1.U; state := outputRequest }
        }.otherwise { outputQuarter := outputQuarter + 1.U; state := outputWrite }
      }
    }
    is(complete)(when(io.cmdResp.fire)(state := idle))
  }
  when(array.io.completed) {
    assert(state === compute || state === drain)
    completed := completed + 1.U
    when(completed + 1.U === count * k) {
      context := 0.U
      row     := 0.U
      state   := store
    }
  }
}
