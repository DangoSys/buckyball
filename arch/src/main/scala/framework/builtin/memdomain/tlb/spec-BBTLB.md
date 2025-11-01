# BBTLB (Translation Lookaside Buffer) Specification

## Overview

BBTLB is a decoupled translation lookaside buffer implementation that accelerates virtual to physical address translation. Inheriting from CoreModule, it provides a TLB interface with exception handling mechanism, supporting page table walk (PTW) and various memory access commands. The module uses parameterized design to support configurable entry count and maximum page size, while integrating complete exception handling flow.

## Interface Design

The module's IO interface contains four main components:
- Request interface (req): Receives TLB request and status information
- Response interface (resp): TLB returns translation result, page fault flags, and access exception information
- Page table walker interface (ptw): PTW interface communicates with memory management unit, handling page table lookup on TLB misses
- Exception handling interface (exp): Exception handling interface manages interrupt signal generation and clearing, supporting both retry and skip flush operation modes

## Internal Implementation

Module internally instantiates a standard TLB module, configured as single-set associative structure (nSets=1, nWays=entries), with instruction TLB feature disabled. Internal TLB's request signal directly connects to input request's tlb_req field, while kill signal is hardwired to false, indicating no support for request cancellation. Page table walker interface communicates with internal TLB through direct connection, while passing request's status information to PTW module to ensure correct permission checking.

## Exception Handling Flow

Exception handling uses interrupt-based mechanism, tracking exception state through RegInit-initialized interrupt register. When valid request with page fault or access exception is detected, module performs corresponding exception checks based on memory command type: for read operations (M_XRD) checks load page fault and access exception, for write operations checks store page fault and access exception. Once exception condition is detected, interrupt signal is set high and maintained until flush operation received and flush signal successfully fires, then cleared.

## SFENCE Operation Support

Module implements complete SFENCE (Supervisor Fence) operation support for TLB flush and synchronization. SFENCE operation trigger condition is any form of flush signal (flush_retry or flush_skip). During SFENCE execution, all related address and ASID fields are set to DontCare, rs1 and rs2 flags are cleared, hv and hg flags are also disabled, indicating this implementation adopts simplified global flush strategy rather than selective flush. Module uses assertion to ensure not receiving retry and skip flush signals simultaneously, guaranteeing operation determinism.

## Parameterized Configuration

Module supports flexible configuration through constructor parameters: entries parameter controls TLB entry count, maxSize parameter defines maximum supported page size. lgMaxSize calculated through log2Ceil, used to determine address width and internal logic precision. This parameterized design enables the module to adapt to different system requirements and performance needs while maintaining interface consistency and implementation reusability.
