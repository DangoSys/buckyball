# BuckyBall Rocket Core Framework

这个目录包含了 BuckyBall 框架中对 Rocket 核心的定制化实现。整个架构建立在 Chipyard/Rocket-chip 框架的基础之上，但为了支持 BuckyBall 的自定义 RoCC 协处理器进行了深度的扩展和修改。BuckyBall 的设计哲学是在保持与上游 Rocket-chip 兼容性的同时，通过并行的类层次结构来实现自己的功能扩展，这样既避免了直接修改上游代码带来的维护问题，又能充分利用 Rocket-chip 的成熟架构。

在 Chipyard 的层次结构中，最顶层是 SoC 子系统，它负责整合多个处理器核心、缓存子系统、互连总线、内存控制器以及各种外设。BuckyBall 通过 `RocketSubsystem.scala` 来定义自己的子系统实现，其中 `RocketSubsystem` 类继承自 Chipyard 的 `BaseSubsystem` 并混入了多个特质来获得必要的功能支持。这些特质包括 `InstantiatesHierarchicalElements` 用于管理层次化组件的实例化，`HasTileNotificationSinks` 和 `HasTileInputConstants` 用于处理 tile 间的通信，`CanHavePeripheryCLINT` 和 `CanHavePeripheryPLIC` 用于中断控制器的支持，以及 `HasPeripheryDebug` 用于调试支持。通过这种多继承的设计模式，BuckyBall 能够复用 Chipyard 的大部分基础设施，同时在需要的地方进行定制化扩展。重要的是，`RocketSubsystem` 还定义了 `RocketTileAttachParamsBB` 来描述如何将 BuckyBall 版本的 Rocket tile 连接到子系统中，这个参数类指定了 tile 的配置参数以及跨时钟域的连接方式。

往下一层是 tile 级别，这里 `RocketTileBB.scala` 定义了单个 Rocket tile 的完整实现。在 Chipyard 的设计中，一个 tile 是一个相对独立的处理单元，包含了处理器核心、L1 指令和数据缓存、可选的向量单元、RoCC 协处理器接口，以及与系统总线的连接接口。`RocketTileBB` 类通过继承 `BaseTile` 获得了基本的 tile 功能，同时混入了多个关键的特质。`SinksExternalInterrupts` 和 `SourcesExternalNotifications` 处理外部中断和通知的接收与发送，`HasLazyRoCCBB` 是 BuckyBall 特有的特质，用于支持 BuckyBall 的 RoCC 协处理器框架，`HasHellaCache` 提供了与 L1 数据缓存的接口，`HasICacheFrontend` 则提供了取指前端的实现。这种多特质组合的设计让 `RocketTileBB` 能够获得所有必要的功能，同时保持代码的模块化和可维护性。特别值得注意的是，`RocketTileBB` 定义了自己的参数类型 `RocketTileParamsBB`，这个参数类包含了 tile 的所有配置信息，包括核心参数、缓存参数、BTB 参数等，并且通过 `InstantiableTileParams` 特质提供了实例化的接口。

在 tile 内部，最核心的组件是处理器核心本身，这由 `RocketCoreBB.scala` 实现。这个文件包含了对原始 Rocket 核心的重新实现，其中 `RocketBB` 类继承自 `CoreModule` 并混入了 `HasRocketCoreParameters` 和 `HasRocketCoreIOBB` 特质。`CoreModule` 提供了核心模块的基本框架，`HasRocketCoreParameters` 提供了各种核心参数的访问接口，而 `HasRocketCoreIOBB` 则定义了 BuckyBall 特有的核心 IO 接口。这个 IO 接口与标准的 Rocket 核心 IO 最大的区别在于使用了 `RoCCCoreIOBB` 而不是标准的 `RoCCCoreIO`，这样就能支持 BuckyBall 特有的 RoCC 接口扩展。在核心的实现中，最关键的修改是指令解码表的处理，原始的 Rocket 核心会根据 `usingRoCC` 参数来决定是否包含 RoCC 指令的解码逻辑，但由于 BuckyBall 使用的是 `BuildRoCCBB` 而不是标准的 `BuildRoCC`，这会导致 `usingRoCC` 返回 false，从而使得 RoCC 指令无法被正确解码。为了解决这个问题，BuckyBall 强制在解码表中包含 `RoCCDecode`，确保自定义指令能够被正确识别和处理。

RoCC 协处理器的支持是通过 `LazyRoCCBB.scala` 来实现的，这个文件定义了 BuckyBall 特有的 RoCC 框架。`HasLazyRoCCBB` 特质是这个框架的核心，它负责管理 RoCC 协处理器的实例化和连接。这个特质会根据 `BuildRoCCBB` 配置来创建相应的 RoCC 实例，并且为每个 RoCC 分配独立的 CSR 地址空间。`HasLazyRoCCModuleBB` 特质则负责在模块级别进行 RoCC 的连接，它实例化了 `RoccCommandRouterBB` 来负责指令的路由。这个路由器会根据指令的 opcode 来决定将指令发送给哪个具体的 RoCC 实例，同时也负责将来自不同 RoCC 的响应进行仲裁后返回给核心。路由器的设计考虑了指令的并发执行和响应的顺序性，确保系统的正确性和性能。

`CSRBB.scala` 包含了对控制状态寄存器子系统的重新实现，这是 BuckyBall 框架中最复杂的部分之一。CSR 子系统负责处理所有的控制状态寄存器访问，包括标准的 RISC-V CSR 以及 BuckyBall 特有的扩展 CSR。这个实现基于原始的 Rocket CSR 实现，但针对 BuckyBall 的需求进行了扩展，支持更灵活的 CSR 地址分配和更复杂的读写逻辑。特别是对于 RoCC 相关的 CSR，BuckyBall 实现了动态的地址分配机制，允许不同的 RoCC 实例拥有独立的 CSR 空间，避免了地址冲突的问题。

`RoCCFragments.scala` 定义了 BuckyBall 版本的 RoCC 接口数据结构，这些结构在保持与标准 RoCC 接口兼容的同时，提供了额外的扩展能力。这包括扩展的命令格式、响应格式，以及额外的控制信号。这些接口定义是整个 BuckyBall RoCC 生态系统的基础，确保了不同组件之间的正确通信。

`Configs.scala` 包含了丰富的配置定义，这些配置类通过 Chipyard 的配置系统来指定各种硬件参数。配置系统使用了函数式编程的思想，通过配置函数的组合来构建复杂的系统配置。BuckyBall 的配置定义了如何将 BuckyBall 特有的组件集成到整个系统中，包括 RoCC 的配置、CSR 的配置、以及各种性能参数的设置。

整个架构最关键的设计挑战在于参数传递和配置的一致性。Chipyard/Rocket-chip 框架广泛使用了 Scala 的隐式参数机制和 `Parameters` 配置系统，这个系统允许配置信息在整个硬件层次结构中进行传递和覆盖。BuckyBall 面临的主要问题是如何确保自己的配置信息（特别是 `BuildRoCCBB` 中定义的 RoCC 配置）能够正确地传递到所有需要这些信息的组件中。由于原始的 Rocket 核心只认识 `BuildRoCC` 而不知道 `BuildRoCCBB` 的存在，BuckyBall 采用了一个巧妙的解决方案：在 `RocketTileBB` 创建 `RocketBB` 核心实例时，动态地修改传递给核心的 `Parameters` 对象，将 `BuildRoCCBB` 中的内容合并到 `BuildRoCC` 中。这样，从核心的视角来看，BuckyBall 的 RoCC 就像是标准的 RoCC 一样，所有基于 `BuildRoCC` 的逻辑都能正常工作，包括 `usingRoCC` 参数的计算、端口数量的计算、以及指令解码表的构建。这种设计既保持了与上游代码的兼容性，又实现了 BuckyBall 所需的功能扩展，体现了软件工程中适配器模式的优雅应用。
