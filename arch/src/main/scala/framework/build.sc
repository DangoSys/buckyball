import mill._
import mill.scalalib._
import $file.device.{build => device_build}
import $file.core.{build => core_build}
import $file.thread.{build => thread_build}
import $file.acc.{build => acc_build}
import $file.ip.{build => ip_build}
import $file.lib.{build => lib_build}
import $file.acc.balldomain.{build => acc_balldomain_build}
import $file.acc.frontend.{build => acc_frontend_build}
import $file.acc.frontend.src.main.scala.boot.{build => acc_frontend_boot_build}
import $file.acc.frontend.src.main.scala.scheduler.{build => acc_frontend_scheduler_build}
import $file.acc.frontend.src.main.scala.scheduler.src.main.scala.bat.{build => acc_frontend_scheduler_bat_build}
import $file.acc.frontend.src.main.scala.scheduler.src.main.scala.rob.global.{build => acc_frontend_scheduler_rob_global_build}
import $file.acc.frontend.src.main.scala.scheduler.src.main.scala.scoreboard.{build => acc_frontend_scheduler_scoreboard_build}
import $file.acc.gpdomain.{build => acc_gpdomain_build}
import $file.acc.gpdomain.src.main.scala.execution.{build => acc_gpdomain_execution_build}
import $file.acc.gpdomain.src.main.scala.loader.{build => acc_gpdomain_loader_build}
import $file.acc.gpdomain.src.main.scala.memory.{build => acc_gpdomain_memory_build}
import $file.acc.memdomain.{build => acc_memdomain_build}
import $file.acc.memdomain.src.main.scala.bank_router.{build => acc_memdomain_bank_router_build}
import $file.acc.memdomain.src.main.scala.prm.{build => acc_memdomain_prm_build}
import $file.acc.memdomain.src.main.scala.tomem.{build => acc_memdomain_tomem_build}
import $file.acc.memdomain.src.main.scala.tomem.src.main.scala.toddr.{build => acc_memdomain_tomem_toddr_build}
import $file.acc.memdomain.src.main.scala.tomem.src.main.scala.tol1.{build => acc_memdomain_tomem_tol1_build}
import $file.core.controller.admission.{build => core_controller_admission_build}
import $file.core.execution.fetch.{build => core_execution_fetch_build}
import $file.core.execution.fpu.{build => core_execution_fpu_build}
import $file.core.execution.instruction.{build => core_execution_instruction_build}
import $file.core.execution.integer.{build => core_execution_integer_build}
import $file.core.execution.muldiv.{build => core_execution_muldiv_build}
import $file.core.execution.privileged.{build => core_execution_privileged_build}
import $file.core.execution.register_file.{build => core_execution_register_file_build}
import $file.core.memory.access.{build => core_memory_access_build}
import $file.core.memory.access.src.main.scala.cache.{build => core_memory_access_cache_build}
import $file.core.memory.access.src.main.scala.mmu.{build => core_memory_access_mmu_build}
import $file.core.memory.access.src.main.scala.preflight.{build => core_memory_access_preflight_build}
import $file.core.memory.home.{build => core_memory_home_build}
import $file.core.tbus.{build => core_tbus_build}
import $file.core.tbus.src.main.scala.router.{build => core_tbus_router_build}
import $file.core.tss.{build => core_tss_build}
import $file.device.ddr.{build => device_ddr_build}
import $file.ip.axi4.{build => ip_axi4_build}
import $file.ip.axis.{build => ip_axis_build}
import $file.ip.bank.{build => ip_bank_build}
import $file.ip.cache.{build => ip_cache_build}
import $file.ip.chi.{build => ip_chi_build}
import $file.ip.coherence.{build => ip_coherence_build}
import $file.ip.interlock.{build => ip_interlock_build}
import $file.ip.mesh_shm.{build => ip_mesh_shm_build}
import $file.ip.spm.{build => ip_spm_build}
import $file.thread.ant.{build => thread_ant_build}
import $file.^.^.^.^.thirdparty.{build => thirdparty_build}

val frameworkRoot: os.Path = millSourcePath
val resourceRoot:  os.Path = millSourcePath / os.up / os.up / "resources"

object library extends SbtModule {
  override def resources           = T.sources(Seq(PathRef(resourceRoot)))
  override def scalaVersion        = "2.13.16"
  override def moduleDeps          = Seq(thirdparty_build.protoJava, thirdparty_build.hardfloat)
  override def ivyDeps             =
    Agg(ivy"org.chipsalliance::chisel:6.7.0", ivy"com.lihaoyi::upickle:3.3.1")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")

  override def sources = T.sources(
    device_build.sources() ++ core_build.sources() ++ thread_build.sources() ++
      acc_build.sources() ++ ip_build.sources() ++ lib_build.sources() ++
      Seq(PathRef(frameworkRoot / os.up / "configs"), PathRef(frameworkRoot / os.up / "sims" / "configs"))
  )

}

object emitters extends SbtModule {
  override def scalaVersion = "2.13.16"
  override def moduleDeps = Seq(library)
  override def ivyDeps = library.ivyDeps()
  override def scalacPluginIvyDeps = library.scalacPluginIvyDeps()
  override def scalacOptions = library.scalacOptions()
  override def sources = T.sources(
    device_build.emitSources() ++ core_build.emitSources() ++ thread_build.emitSources() ++
      acc_build.emitSources() ++ ip_build.emitSources() ++ lib_build.emitSources()
  )
}

object acc extends Module {
  object balldomain extends acc_balldomain_build.Emitter {
    override def framework = library
  }
  object frontend extends acc_frontend_build.Emitter {
    override def framework = library
    object boot extends acc_frontend_boot_build.Emitter {
      override def framework = library
    }
    object scheduler extends acc_frontend_scheduler_build.Emitter {
      override def framework = library
      object bat extends acc_frontend_scheduler_bat_build.Emitter {
        override def framework = library
      }
      object rob extends Module {
        object global extends acc_frontend_scheduler_rob_global_build.Emitter {
          override def framework = library
        }
      }
      object scoreboard extends acc_frontend_scheduler_scoreboard_build.Emitter {
        override def framework = library
      }
    }
  }
  object gpdomain extends acc_gpdomain_build.Emitter {
    override def framework = library
    object execution extends acc_gpdomain_execution_build.Emitter {
      override def framework = library
    }
    object loader extends acc_gpdomain_loader_build.Emitter {
      override def framework = library
    }
    object memory extends acc_gpdomain_memory_build.Emitter {
      override def framework = library
    }
  }
  object memdomain extends acc_memdomain_build.Emitter {
    override def framework = library
    object bank_router extends acc_memdomain_bank_router_build.Emitter {
      override def framework = library
    }
    object prm extends acc_memdomain_prm_build.Emitter {
      override def framework = library
    }
    object tomem extends acc_memdomain_tomem_build.Emitter {
      override def framework = library
      object toddr extends acc_memdomain_tomem_toddr_build.Emitter {
        override def framework = library
      }
      object tol1 extends acc_memdomain_tomem_tol1_build.Emitter {
        override def framework = library
      }
    }
  }
}
object core extends Module {
  object controller extends Module {
    object admission extends core_controller_admission_build.Emitter {
      override def framework = library
    }
  }
  object execution extends Module {
    object fetch extends core_execution_fetch_build.Emitter {
      override def framework = library
    }
    object fpu extends core_execution_fpu_build.Emitter {
      override def framework = library
    }
    object instruction extends core_execution_instruction_build.Emitter {
      override def framework = library
    }
    object integer extends core_execution_integer_build.Emitter {
      override def framework = library
    }
    object muldiv extends core_execution_muldiv_build.Emitter {
      override def framework = library
    }
    object privileged extends core_execution_privileged_build.Emitter {
      override def framework = library
    }
    object register_file extends core_execution_register_file_build.Emitter {
      override def framework = library
    }
  }
  object memory extends Module {
    object access extends core_memory_access_build.Emitter {
      override def framework = library
      object cache extends core_memory_access_cache_build.Emitter {
        override def framework = library
      }
      object mmu extends core_memory_access_mmu_build.Emitter {
        override def framework = library
      }
      object preflight extends core_memory_access_preflight_build.Emitter {
        override def framework = library
      }
    }
    object home extends core_memory_home_build.Emitter {
      override def framework = library
    }
  }
  object tbus extends core_tbus_build.Emitter {
    override def framework = library
    object router extends core_tbus_router_build.Emitter {
      override def framework = library
    }
  }
  object tss extends core_tss_build.Emitter {
    override def framework = library
  }
}
object device extends Module {
  object ddr extends device_ddr_build.Emitter {
    override def framework = library
  }
}
object ip extends Module {
  object axi4 extends ip_axi4_build.Emitter {
    override def framework = library
  }
  object axis extends ip_axis_build.Emitter {
    override def framework = library
  }
  object bank extends ip_bank_build.Emitter {
    override def framework = library
  }
  object cache extends ip_cache_build.Emitter {
    override def framework = library
  }
  object chi extends ip_chi_build.Emitter {
    override def framework = library
  }
  object coherence extends ip_coherence_build.Emitter {
    override def framework = library
  }
  object interlock extends ip_interlock_build.Emitter {
    override def framework = library
  }
  object mesh_shm extends ip_mesh_shm_build.Emitter {
    override def framework = library
  }
  object spm extends ip_spm_build.Emitter {
    override def framework = library
  }
}
object thread extends Module {
  object ant extends thread_ant_build.Emitter {
    override def framework = library
  }
}
