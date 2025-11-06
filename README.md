<p align="center">
    <img src="https://github.com/DangoSys/buckyball/raw/main/docs/img/logo.png" width = "100%" height = "70%">
</p>

<div align="center" style="margin-top: -10pt;">

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://app.devin.ai/wiki/DangoSys/buckyball)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/DangoSys/buckyball)
[![Document](https://github.com/DangoSys/buckyball/actions/workflows/doc.yml/badge.svg?branch=main)](https://dangosys.github.io/buckyball)
[![Buckyball CI](https://github.com/DangoSys/buckyball/actions/workflows/test.yml/badge.svg)](https://github.com/DangoSys/buckyball/actions/workflows/test.yml)

</div>

# BuckyBall

BuckyBall is a scalable framework for Domain Specific Architecture, built on RISC-V architecture and optimized for high-performance computing and machine learning accelerator design.

## Project Overview

The BuckyBall framework provides a complete hardware design, simulation verification, and software development toolchain, supporting the full development process from RTL design to system-level verification. The framework adopts a modular design that supports flexible configuration and extension, suitable for various specialized computing scenarios.

## Quick Start

### Environment Dependencies

Before getting started, please ensure your system meets the following dependency requirements:

**Required Software**:
- Anaconda/Miniconda (Python environment management)
- Ninja Build System
- GTKWave (waveform viewer)
- Bash Shell environment (doesn't need to be the primary shell)

**Installing Dependencies**:
```bash
# Install Anaconda
# Download from: https://www.anaconda.com/download/

# Install system tools
sudo apt install ninja-build gtkwave

# Optional: FireSim passwordless configuration
# Add to /etc/sudoers: user_name ALL=(ALL) NOPASSWD:ALL
```

### Source Build

**1. Clone Repository**
```bash
git clone https://github.com/DangoSys/buckyball.git
cd buckyball
```

**2. Initialize Environment**
```bash
./scripts/init.sh
```
*Note: Initialization takes approximately 3 hours, including dependency downloads and compilation*

**3. Environment Activation**
```bash
source buckyball/env.sh
```

### Verify Installation

Run Verilator simulation test to verify installation:
```bash
bbdev verilator --run '--jobs 16 --binary ctest_vecunit_matmul_ones_singlecore-baremetal --batch'
```

### Docker Quick Experience

**Notice**:
- Docker images are provided only for specific release versions.
- Docker image may not be the latest version, source build is recommended.

**Pull Docker Image**
```bash
docker pull ghcr.io/dangosys/buckyball:latest
```

**Run Docker Container**
```bash
docker run -it ghcr.io/dangosys/buckyball:latest
# Activate environment
source buckyball/env.sh
# Run Verilator simulation test
bbdev verilator --run '--jobs 16 --binary ctest_vecunit_matmul_ones_singlecore-baremetal --batch'
```

## Additional Resources

You can learn more from [DeepWiki](https://deepwiki.com/DangoSys/buckyball) and [Zread](https://zread.ai/DangoSys/buckyball)

## Community

Join our discussion on [Slack](https://dangosys-buckyball.slack.com/)

## Contributors
Thank you for considering contributing to Buckyball!

<a href="https://github.com/DangoSys/buckyball/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=DangoSys/buckyball" />
</a>
