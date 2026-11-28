import mill._
import mill.scalalib._
import $file.device.{build => device_build}
import $file.tlp.memory.{build => hart_dma_build}
import $file.device.peripherals.scu.{build => device_scu_build}
import $file.device.peripherals.trace.{build => device_trace_build}
import $file.device.peripherals.plic.{build => device_plic_build}
import $file.device.peripherals.clint.{build => device_clint_build}
import $file.device.peripherals.bootrom.{build => device_bootrom_build}
import $file.hart.{build => hart_build}
import $file.hart.cbus.{build => hart_cbus_build}
import $file.thread.{build => thread_build}
import $file.generic.{build => generic_build}
import $file.ball.{build => ball_build}
import $file.ilp.{build => ilp_build}
import $file.mlp.{build => mlp_build}
import $file.mlp.fetch.{build => mlp_fetch_build}
import $file.tlp.{build => tlp_build}
import $file.custom.{build => custom_build}
import $file.custom.frontend.{build => custom_frontend_build}
import $file.thread.acc.frontend.{build => thread_acc_frontend_build}
import $file.custom.frontend.src.main.scala.boot.{build => custom_frontend_boot_build}
import $file.thread.acc.frontend.src.main.scala.boot.{build => thread_acc_frontend_boot_build}
import $file.custom.frontend.src.main.scala.scheduler.{build => custom_frontend_scheduler_build}
import $file.thread.acc.frontend.src.main.scala.scheduler.{build => thread_acc_frontend_scheduler_build}
import $file.custom.frontend.src.main.scala.scheduler.src.main.scala.bat.{build => custom_frontend_scheduler_bat_build}
import $file.thread.acc.frontend.src.main.scala.scheduler.src.main.scala.bat.{build => thread_acc_frontend_scheduler_bat_build}
import $file.custom.frontend.src.main.scala.scheduler.src.main.scala.rob.global.{build => custom_frontend_scheduler_rob_global_build}
import $file.thread.acc.frontend.src.main.scala.scheduler.src.main.scala.rob.global.{build => thread_acc_frontend_scheduler_rob_global_build}
import $file.custom.frontend.src.main.scala.scheduler.src.main.scala.scoreboard.{build => custom_frontend_scheduler_scoreboard_build}
import $file.thread.acc.frontend.src.main.scala.scheduler.src.main.scala.scoreboard.{build => thread_acc_frontend_scheduler_scoreboard_build}
import $file.custom.gpdomain.{build => custom_gpdomain_build}
import $file.thread.acc.gpdomain.{build => thread_acc_gpdomain_build}
import $file.custom.gpdomain.src.main.scala.execution.{build => custom_gpdomain_execution_build}
import $file.thread.acc.gpdomain.src.main.scala.execution.{build => thread_acc_gpdomain_execution_build}
import $file.custom.gpdomain.src.main.scala.loader.{build => custom_gpdomain_loader_build}
import $file.thread.acc.gpdomain.src.main.scala.loader.{build => thread_acc_gpdomain_loader_build}
import $file.custom.gpdomain.src.main.scala.memory.{build => custom_gpdomain_memory_build}
import $file.thread.acc.gpdomain.src.main.scala.memory.{build => thread_acc_gpdomain_memory_build}
import $file.custom.memdomain.{build => custom_memdomain_build}
import $file.thread.acc.memdomain.{build => thread_acc_memdomain_build}
import $file.thread.acc.memdomain.src.main.scala.tomem.{build => thread_tomem_build}
import $file.custom.memdomain.src.main.scala.bank_router.{build => custom_memdomain_bank_router_build}
import $file.thread.acc.memdomain.src.main.scala.bank_router.{build => thread_acc_memdomain_bank_router_build}
import $file.custom.memdomain.src.main.scala.prm.{build => custom_memdomain_prm_build}
import $file.thread.acc.memdomain.src.main.scala.prm.{build => thread_acc_memdomain_prm_build}
import $file.custom.memdomain.src.main.scala.tomem.{build => custom_memdomain_tomem_build}
import $file.custom.memdomain.src.main.scala.tomem.src.main.scala.toddr.{build => custom_memdomain_tomem_toddr_build}
import $file.custom.memdomain.src.main.scala.tomem.src.main.scala.tol1.{build => custom_memdomain_tomem_tol1_build}
import $file.tlp.controller.admission.{build => tlp_controller_admission_build}
import $file.hart.execution.fpu.{build => hart_execution_fpu_build}
import $file.hart.execution.instruction.{build => hart_execution_instruction_build}
import $file.hart.execution.integer.{build => hart_execution_integer_build}
import $file.hart.execution.muldiv.{build => hart_execution_muldiv_build}
import $file.hart.execution.privileged.{build => hart_execution_privileged_build}
import $file.hart.execution.register_file.{build => hart_execution_register_file_build}
import $file.hart.memory.access.{build => hart_memory_access_build}
import $file.hart.memory.{build => hart_memory_build}
import $file.hart.memory.access.src.main.scala.cache.{build => hart_memory_access_cache_build}
import $file.hart.memory.access.src.main.scala.mmu.{build => hart_memory_access_mmu_build}
import $file.hart.memory.access.src.main.scala.preflight.{build => hart_memory_access_preflight_build}
import $file.hart.memory.home.{build => hart_memory_home_build}
import $file.tlp.tbus.{build => tlp_tbus_build}
import $file.tlp.tbus.src.main.scala.router.{build => tlp_tbus_router_build}
import $file.tlp.tss.{build => tlp_tss_build}
import $file.tlp.shm.{build => tlp_shm_build}
import $file.generic.ddr.{build => generic_ddr_build}
import $file.generic.axi4.{build => generic_axi4_build}
import $file.generic.axis.{build => generic_axis_build}
import $file.generic.bank.{build => generic_bank_build}
import $file.generic.cache.{build => generic_cache_build}
import $file.generic.chi.{build => generic_chi_build}
import $file.generic.coherence.{build => generic_coherence_build}
import $file.generic.interlock.{build => generic_interlock_build}
import $file.generic.spm.{build => generic_spm_build}
import $file.thread.ant.{build => thread_ant_build}
import $file.^.^.^.thirdparty.{build => thirdparty_build}

val frameworkRoot: os.Path = millSourcePath
val resourceRoot:  os.Path = millSourcePath / os.up / "resources"

// One module per abstraction, each on the generic library alone: a reference
// from one abstraction to another does not compile. Only the simulation tops
// and the designs assemble them.
trait Layer extends SbtModule {
  override def scalaVersion        = "2.13.16"
  override def ivyDeps             =
    Agg(ivy"org.chipsalliance::chisel:6.7.0", ivy"com.lihaoyi::upickle:3.3.1")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")
}

object layers extends Module {

  object generic extends Layer {
    override def resources  = T.sources(Seq(PathRef(resourceRoot)))
    override def moduleDeps = Seq(thirdparty_build.protoJava, thirdparty_build.hardfloat)
    override def sources    = T.sources(generic_build.sources())
  }

  object ball extends Layer {
    override def moduleDeps = Seq(generic)
    override def sources    = T.sources(ball_build.sources())
  }

  object thread extends Layer {
    override def moduleDeps = Seq(generic)
    override def sources    = T.sources(thread_build.sources())
  }

  object hart extends Layer {
    override def moduleDeps = Seq(generic)
    override def sources    = T.sources(hart_build.sources())
  }

  object device extends Layer {
    override def moduleDeps = Seq(generic)
    override def sources    = T.sources(device_build.sources())
  }
  // The CPU's four optimization abstractions. Each plugs into the base hart and
  // none sees another; the base hart sees none of them, so a design composes a
  // CPU from the base and the ones it wants, and the framework fixes no choice.
  object ilp extends Layer {
    override def moduleDeps = Seq(generic, hart)
    override def sources    = T.sources(ilp_build.sources())
  }
  object mlp extends Layer {
    override def moduleDeps = Seq(generic, hart)
    override def sources    = T.sources(mlp_build.sources())
  }
  object tlp extends Layer {
    override def moduleDeps = Seq(generic, hart)
    override def sources    = T.sources(tlp_build.sources())
  }
  object custom extends Layer {
    override def moduleDeps = Seq(generic, hart)
    override def sources    = T.sources(custom_build.sources())
  }

  object sims extends Layer {
    override def moduleDeps = Seq(generic, ball, thread, hart, device, ilp, mlp, tlp, custom)
    override def sources    = T.sources(device_build.simSources() ++ hart_build.simSources() ++ thread_build.simSources())
  }

}

// The whole framework, for the designs.
object library extends Layer {
  override def moduleDeps = Seq(
    layers.generic,
    layers.ball,
    layers.thread,
    layers.hart,
    layers.device,
    layers.ilp,
    layers.mlp,
    layers.tlp,
    layers.custom,
    layers.sims
  )
  override def sources    = T.sources(Seq.empty[PathRef])
}

object emitters extends SbtModule {
  override def scalaVersion = "2.13.16"
  override def moduleDeps = Seq(library)
  override def ivyDeps = library.ivyDeps()
  override def scalacPluginIvyDeps = library.scalacPluginIvyDeps()
  override def scalacOptions = library.scalacOptions()
  override def sources = T.sources(
    device_build.emitSources() ++ hart_build.emitSources() ++ thread_build.emitSources() ++
      ball_build.emitSources() ++ generic_build.emitSources() ++ ilp_build.emitSources() ++
      mlp_build.emitSources() ++ tlp_build.emitSources() ++ custom_build.emitSources()
  )
}

object ball extends ball_build.Emitter {
  override def framework = library
}
object device extends Module {
  object scu extends device_scu_build.Emitter {
    override def framework = layers.device
  }
  object trace extends device_trace_build.Emitter {
    override def framework = layers.device
  }
  object plic extends device_plic_build.Emitter {
    override def framework = layers.device
  }
  object clint extends device_clint_build.Emitter {
    override def framework = layers.device
  }
  object bootrom extends device_bootrom_build.Emitter {
    override def framework = layers.device
  }
}

object hart extends Module {
  object cbus extends hart_cbus_build.Emitter {
    override def framework = layers.sims
  }
  object execution extends Module {
    object fpu extends hart_execution_fpu_build.Emitter {
      override def framework = library
    }
    object instruction extends hart_execution_instruction_build.Emitter {
      override def framework = library
    }
    object integer extends hart_execution_integer_build.Emitter {
      override def framework = library
    }
    object muldiv extends hart_execution_muldiv_build.Emitter {
      override def framework = library
    }
    object privileged extends hart_execution_privileged_build.Emitter {
      override def framework = library
    }
    object register_file extends hart_execution_register_file_build.Emitter {
      override def framework = library
    }
  }
  object memory extends hart_memory_build.Emitter {
    override def framework = library
    object access extends hart_memory_access_build.Emitter {
      override def framework = library
      object cache extends hart_memory_access_cache_build.Emitter {
        override def framework = library
      }
      object mmu extends hart_memory_access_mmu_build.Emitter {
        override def framework = library
      }
      object preflight extends hart_memory_access_preflight_build.Emitter {
        override def framework = library
      }
    }
    object home extends hart_memory_home_build.Emitter {
      override def framework = library
    }
  }
}
object custom extends Module {
  object frontend extends custom_frontend_build.Emitter {
    override def framework = library
    object boot extends custom_frontend_boot_build.Emitter {
      override def framework = library
    }
    object scheduler extends custom_frontend_scheduler_build.Emitter {
      override def framework = library
      object bat extends custom_frontend_scheduler_bat_build.Emitter {
        override def framework = library
      }
      object rob extends Module {
        object global extends custom_frontend_scheduler_rob_global_build.Emitter {
          override def framework = library
        }
      }
      object scoreboard extends custom_frontend_scheduler_scoreboard_build.Emitter {
        override def framework = library
      }
    }
  }
  object gpdomain extends custom_gpdomain_build.Emitter {
    override def framework = library
    object execution extends custom_gpdomain_execution_build.Emitter {
      override def framework = library
    }
    object loader extends custom_gpdomain_loader_build.Emitter {
      override def framework = library
    }
    object memory extends custom_gpdomain_memory_build.Emitter {
      override def framework = library
    }
  }
  object memdomain extends custom_memdomain_build.Emitter {
    override def framework = library
    object bank_router extends custom_memdomain_bank_router_build.Emitter {
      override def framework = library
    }
    object prm extends custom_memdomain_prm_build.Emitter {
      override def framework = library
    }
    object tomem extends custom_memdomain_tomem_build.Emitter {
      override def framework = library
      object toddr extends custom_memdomain_tomem_toddr_build.Emitter {
        override def framework = library
      }
      object tol1 extends custom_memdomain_tomem_tol1_build.Emitter {
        override def framework = library
      }
    }
  }
}
object tlp extends Module {
  object memory extends hart_dma_build.Emitter {
    override def framework = library
  }
  object controller extends Module {
    object admission extends tlp_controller_admission_build.Emitter {
      override def framework = library
    }
  }
  object tbus extends tlp_tbus_build.Emitter {
    override def framework = library
    object router extends tlp_tbus_router_build.Emitter {
      override def framework = library
    }
  }
  object tss extends tlp_tss_build.Emitter {
    override def framework = library
  }
  object shm extends tlp_shm_build.Emitter {
    override def framework = library
  }
}
object mlp extends Module {
  object fetch extends mlp_fetch_build.Emitter {
    override def framework = library
  }
}
object generic extends Module {
  object ddr extends generic_ddr_build.Emitter {
    override def framework = library
  }
  object axi4 extends generic_axi4_build.Emitter {
    override def framework = library
  }
  object axis extends generic_axis_build.Emitter {
    override def framework = library
  }
  object bank extends generic_bank_build.Emitter {
    override def framework = library
  }
  object cache extends generic_cache_build.Emitter {
    override def framework = library
  }
  object chi extends generic_chi_build.Emitter {
    override def framework = library
  }
  object coherence extends generic_coherence_build.Emitter {
    override def framework = library
  }
  object interlock extends generic_interlock_build.Emitter {
    override def framework = library
  }
  object spm extends generic_spm_build.Emitter {
    override def framework = library
  }
}
object thread extends Module {
  object ant extends thread_ant_build.Emitter {
    override def framework = library
  }
  object acc extends Module {
    object frontend extends thread_acc_frontend_build.Emitter {
      override def framework = library
      object boot extends thread_acc_frontend_boot_build.Emitter {
        override def framework = library
      }
      object scheduler extends thread_acc_frontend_scheduler_build.Emitter {
        override def framework = library
        object bat extends thread_acc_frontend_scheduler_bat_build.Emitter {
          override def framework = library
        }
        object rob extends Module {
          object global extends thread_acc_frontend_scheduler_rob_global_build.Emitter {
            override def framework = library
          }
        }
        object scoreboard extends thread_acc_frontend_scheduler_scoreboard_build.Emitter {
          override def framework = library
        }
      }
    }
    object gpdomain extends thread_acc_gpdomain_build.Emitter {
      override def framework = library
      object execution extends thread_acc_gpdomain_execution_build.Emitter {
        override def framework = library
      }
      object loader extends thread_acc_gpdomain_loader_build.Emitter {
        override def framework = library
      }
      object memory extends thread_acc_gpdomain_memory_build.Emitter {
        override def framework = library
      }
    }
    object memdomain extends Module {
      object tomem extends thread_tomem_build.Emitter {
        override def framework = library
      }
      object bank_router extends thread_acc_memdomain_bank_router_build.Emitter {
        override def framework = library
      }
      object prm extends thread_acc_memdomain_prm_build.Emitter {
        override def framework = library
      }
    }
  }
}
