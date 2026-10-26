import mill._
import $file.fetch.{build => fetch_build}
import $file.fpu.{build => fpu_build}
import $file.instruction.{build => instruction_build}
import $file.integer.{build => integer_build}
import $file.muldiv.{build => muldiv_build}
import $file.privileged.{build => privileged_build}
import $file.register_file.{build => register_file_build}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("Execution.scala")),
    PathRef(millSourcePath / os.RelPath("configs/ExecutionParam.scala")),
    PathRef(millSourcePath / os.RelPath("pipeline/Sequencer.scala"))
  ) ++
  fetch_build.sources() ++
  fpu_build.sources() ++
  instruction_build.sources() ++
  integer_build.sources() ++
  muldiv_build.sources() ++
  privileged_build.sources() ++
  register_file_build.sources()
)

def emitSources = T.sources(
  fetch_build.emitSources() ++
  fpu_build.emitSources() ++
  instruction_build.emitSources() ++
  integer_build.emitSources() ++
  muldiv_build.emitSources() ++
  privileged_build.emitSources() ++
  register_file_build.emitSources()
)
