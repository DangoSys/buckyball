<p align="center">
    <img src="https://raw.githubusercontent.com/DangoSys/document/refs/heads/main/resource/images/logo-long.png" width = "100%" height = "70%">
</p>

<div align="center" style="margin-top: -10pt;">

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/DangoSys/buckyball)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/DangoSys/buckyball)
[![Document](https://img.shields.io/badge/documents-online-30c452?style=plastic&logo=gitbook)](https://docs.buckyball.tech/zh/%E6%95%99%E7%A8%8B/%E4%BB%93%E5%BA%93%E6%A6%82%E8%A7%88%E4%B8%8E%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA)
[![buckyball CI](https://github.com/DangoSys/buckyball/actions/workflows/test.yml/badge.svg)](https://github.com/DangoSys/buckyball/actions/workflows/test.yml)

</div>


> [!NOTE]
> ### What is Buckyball?
> Buckyball is an open-source architectural framework for building domain-specific architectures (DSAs). A DSA is a hardware architecture optimized for a particular class of workloads, such as machine learning, graph processing, signal processing, or scientific computing. Examples include systolic-array accelerators designed for Transformer models and other specialized computing engines. Rather than defining a single fixed accelerator, Buckyball provides a unified architectural abstraction, standardized interfaces, and system-level infrastructure that allow diverse DSAs to be integrated into a common platform. This enables different accelerators to be developed, deployed, and executed within the same system while sharing a consistent software and hardware environment.
> ### Why We Need Such a Framework?
> **Every DSA project comes with a substantial amount of system re-engineering.** Beyond the accelerator itself, developers must build instruction interfaces, scheduling mechanisms, memory systems, software stacks, verification flows, and simulation infrastructure. Much of this work is common across projects, yet it is repeatedly implemented from scratch. As a result, valuable engineering effort is spent reinventing shared infrastructure, often limiting the scalability, portability, and generality of DSA designs.
> Buckyball addresses this challenge by providing a reusable and extensible open-source foundation that captures the common infrastructure shared by DSA projects. By abstracting away repetitive system engineering, it allows researchers and developers to focus on the design and optimization of their own accelerators instead of rebuilding the surrounding ecosystem from scratch. The framework also makes it easier to publish, evaluate, and integrate new DSAs, enabling multiple accelerators to coexist and collaborate within a unified system.

> [!IMPORTANT]
> **Our goal is to make DSA development faster, more reusable, and easier to share.**

## Quick Start

### Installation in Nix
We use Nix Flake as our main build system. If you have not installed nix, install it following the [guide](https://nix.dev/manual/nix/2.28/installation/installing-binary.html), and enable flake following the [wiki](https://nixos.wiki/wiki/Flakes#Enable_flakes). Or you can try the [installer](https://github.com/DeterminateSystems/nix-installer) provided by Determinate Systems, which enables flake by default.


**1. Clone Repository**

```bash
git clone https://github.com/DangoSys/buckyball.git
```

**2. Initialize Environment**
```bash
cd buckyball
./scripts/nix/build-all.sh
```

After the first time installation, you can enter the environment anytime by running:

```bash
nix develop
```

**3. Verify Installation**

Run Verilator simulation test to verify installation:

```bash
bbdev verilator --run '--jobs 16 --binary vecunit_matmul_ones-singlecore-baremetal --config sims.verilator.BuckyballToyVerilatorConfig --batch'
```

**4. Try faster simulation using bebop**

```bash
bbdev bebop-verilator --run '--binary vecunit_matmul_ones-singlecore-baremetal --batch --config sims.verilator.BuckyballToyVerilatorConfig --itrace --mtrace --pmctrace --ctrace --banktrace'
```

## Tutorial
You can start to learn ball and blink from [here](https://docs.buckyball.tech/zh/%E6%95%99%E7%A8%8B/%E4%BB%80%E4%B9%88%E6%98%AFBall+%26+%E5%A6%82%E4%BD%95%E5%86%99%E4%B8%80%E4%B8%AABall)

## Additional Resources

You can learn more from [DeepWiki](https://deepwiki.com/DangoSys/buckyball) and [Zread](https://zread.ai/DangoSys/buckyball)


## Contributors
Thank you for considering contributing to buckyball!

<a href="https://github.com/DangoSys/buckyball/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=DangoSys/buckyball" />
</a>
