import mill._
import $file.axi4.{build => axi4_build}
import $file.axis.{build => axis_build}
import $file.bank.{build => bank_build}
import $file.cache.{build => cache_build}
import $file.chi.{build => chi_build}
import $file.coherence.{build => coherence_build}
import $file.interlock.{build => interlock_build}
import $file.mesh_shm.{build => mesh_shm_build}
import $file.spm.{build => spm_build}

def sources = T.sources(
  axi4_build.sources() ++
  axis_build.sources() ++
  bank_build.sources() ++
  cache_build.sources() ++
  chi_build.sources() ++
  coherence_build.sources() ++
  interlock_build.sources() ++
  mesh_shm_build.sources() ++
  spm_build.sources()
)

def emitSources = T.sources(
  axi4_build.emitSources() ++
  axis_build.emitSources() ++
  bank_build.emitSources() ++
  cache_build.emitSources() ++
  chi_build.emitSources() ++
  coherence_build.emitSources() ++
  interlock_build.emitSources() ++
  mesh_shm_build.emitSources() ++
  spm_build.emitSources()
)
