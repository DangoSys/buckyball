package sims.base

import org.chipsalliance.cde.config.Config

// =============================================================================
// BuckyballBaseConfig: minimal shared config for all Buckyball simulation platforms
// Does NOT inherit chipyard.config.AbstractConfig to avoid generating unused
// IO ports and test infrastructure (SerialTL/SimTSI, UART, Debug, ClockTap).
//
// This base config provides:
//   - Minimal harness/IO binders (no SimTSI, no UART adapter, no Debug)
//   - Memory system (coherence, buses, scratchpad)
//   - Clocking infrastructure
//   - Disabled unused features
//
// Platform-specific configs (P2E, Verilator) should extend this and add:
//   - Platform-specific harness binders (e.g., WithSCU, WithBBSimMem)
//   - Memory port configuration (e.g., WithP2EDDR4MemPort, WithDefaultMemPort)
//   - BootROM configuration
//   - Tile configuration (e.g., WithNToyTiles)
// =============================================================================
class BuckyballBaseConfig
    extends Config(
      // ================================================
      //   Harness Binders (minimal - no SimTSI/UART/Debug)
      // ================================================
      new chipyard.harness.WithSerialTLTiedOff ++    // tie-off SerialTL if present
        new chipyard.harness.WithTieOffInterrupts ++ // tie-off interrupt ports if present
        new chipyard.harness.WithTieOffL2FBusAXI ++  // tie-off external AXI4 master if present
        new chipyard.harness.WithClockFromHarness ++ // ChipTop clocks driven by harnessClockInstantiator
        new chipyard.harness.WithResetFromHarness ++ // reset controlled by harness
        new chipyard.harness.WithAbsoluteFreqHarnessClockInstantiator ++

        // ================================================
        //   IO Binders (only what's needed)
        // ================================================
        new chipyard.iobinders.WithAXI4MemPunchthrough ++ // expose mem_axi4 to top
        new chipyard.iobinders.WithExtInterruptIOCells ++ // handle external interrupt ports
        new chipyard.iobinders.WithNMITiedOff ++

        // ================================================
        //   Disable unused features
        // ================================================
        new chipyard.config.WithNoDebug ++         // no JTAG/Debug module
        new chipyard.config.WithNoUART ++          // no 16550 UART
        new chipyard.config.WithNoClockTap ++      // no clock_tap pin
        new testchipip.boot.WithNoCustomBootPin ++ // no custom_boot pin
        new testchipip.serdes.WithNoSerialTL ++    // no serial_tl interface

        // ================================================
        //   External Memory (no MMIO/Slave port)
        // ================================================
        new freechips.rocketchip.subsystem.WithNMemoryChannels(1) ++
        new freechips.rocketchip.subsystem.WithNoMMIOPort ++  // no top-level MMIO master port
        new freechips.rocketchip.subsystem.WithNoSlavePort ++ // no top-level slave port

        // ================================================
        //   Interrupts
        // ================================================
        new freechips.rocketchip.subsystem.WithNExtTopInterrupts(0) ++

        // ================================================
        //   Memory system
        // ================================================
        new freechips.rocketchip.subsystem.WithDTS("ucb-bar,buckyball", Nil) ++
        new chipyard.config.WithBootROM ++
        new testchipip.soc.WithMbusScratchpad(base = 0x08000000, size = 64 * 1024) ++ // 64 KiB on-chip scratchpad
        new freechips.rocketchip.subsystem.WithInclusiveCache ++
        new freechips.rocketchip.subsystem.WithCoherentBusTopology ++

        // ================================================
        //   Clocking
        // ================================================
        new chipyard.clocking.WithPassthroughClockGenerator ++
        new freechips.rocketchip.subsystem.WithDontDriveBusClocksFromSBus ++
        new freechips.rocketchip.subsystem.WithClockGateModel ++
        new chipyard.clocking.WithClockGroupsCombinedByName((
          "uncore",
          Seq("sbus", "mbus", "pbus", "fbus", "cbus", "obus", "implicit"),
          Seq("tile")
        )) ++
        new chipyard.config.WithPeripheryBusFrequency(500.0) ++
        new chipyard.config.WithMemoryBusFrequency(500.0) ++
        new chipyard.config.WithControlBusFrequency(500.0) ++
        new chipyard.config.WithSystemBusFrequency(500.0) ++
        new chipyard.config.WithFrontBusFrequency(500.0) ++
        new chipyard.config.WithOffchipBusFrequency(500.0) ++
        new chipyard.config.WithInheritBusFrequencyAssignments ++
        new chipyard.config.WithNoSubsystemClockIO ++

        // ================================================
        //   Base settings
        // ================================================
        new freechips.rocketchip.system.BaseConfig
    )
