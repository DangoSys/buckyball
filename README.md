<p align="center">
    <img src="https://github.com/DangoSys/buckyball/raw/main/docs/img/buckyball.png" width = "100%" height = "70%">
</p>

<div align="center" style="margin-top: -10pt;">

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/DangoSys/buckyball)
[![Ask Zread](https://img.shields.io/badge/Ask_Zread-8A2BE2)](https://zread.ai/DangoSys/buckyball)
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
*Note: Initialization takes approximately 2 hours, including dependency downloads and compilation*

**3. Environment Activation**
```bash
source buckyball/env.sh
```

### Verify Installation

Run Verilator simulation test to verify installation:
```bash
bbdev verilator --run '--jobs 16 --binary ctest_mvin_mvout_alternate_test_singlecore-baremetal --batch'
```

### Docker Quick Experience (Not updated for a while)

```bash
docker pull ghcr.io/dangosys/buckyball:latest
```
*Note: Docker image may not be the latest version, source build is recommended*

## Additional Resources

You can learn more from [DeepWiki](https://deepwiki.com/DangoSys/buckyball) and [Zread](https://zread.ai/DangoSys/buckyball)

## Community

Join our community discussion on [Slack](https://buckyballhq.slack.com/)
