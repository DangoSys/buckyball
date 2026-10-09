# Standalone FPGA platforms

These examples provide board peripherals, firmware and implementation scripts
alongside an explicitly pinned Buckyball core configuration. They do not change
the default chip configurations or the bbdev/P2E build flow.

| Platform | Device | Memory and software |
|---|---|---|
| [Arty A7-100T](arty-a7/README.md) | `xc7a100tcsg324-1` | BRAM, serial program loader, RV64 bare-metal CPU and three operators |

Read each platform's validation record before using a bitstream. RTL simulation
and routed timing closure are distinct from physical board acceptance.
