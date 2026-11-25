import mill._
import $file.acc.{build => acc_build}
import $file.arbiter.{build => arbiter_build}
import $file.arith.{build => arith_build}
import $file.axi4.{build => axi4_build}
import $file.axis.{build => axis_build}
import $file.bank.{build => bank_build}
import $file.cache.{build => cache_build}
import $file.chi.{build => chi_build}
import $file.coherence.{build => coherence_build}
import $file.contract.{build => contract_build}
import $file.ddr.{build => ddr_build}
import $file.dma.{build => dma_build}
import $file.dpi.{build => dpi_build}
import $file.hash.{build => hash_build}
import $file.interlock.{build => interlock_build}
import $file.queue.{build => queue_build}
import $file.spm.{build => spm_build}

def sources = T.sources(
  acc_build.sources() ++
  arbiter_build.sources() ++
  arith_build.sources() ++
  axi4_build.sources() ++
  axis_build.sources() ++
  bank_build.sources() ++
  cache_build.sources() ++
  chi_build.sources() ++
  coherence_build.sources() ++
  contract_build.sources() ++
  ddr_build.sources() ++
  dma_build.sources() ++
  dpi_build.sources() ++
  hash_build.sources() ++
  interlock_build.sources() ++
  queue_build.sources() ++
  spm_build.sources()
)

def emitSources = T.sources(
  arbiter_build.emitSources() ++
  arith_build.emitSources() ++
  axi4_build.emitSources() ++
  axis_build.emitSources() ++
  bank_build.emitSources() ++
  cache_build.emitSources() ++
  chi_build.emitSources() ++
  coherence_build.emitSources() ++
  ddr_build.emitSources() ++
  dpi_build.emitSources() ++
  hash_build.emitSources() ++
  interlock_build.emitSources() ++
  queue_build.emitSources() ++
  spm_build.emitSources()
)
