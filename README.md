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

BuckyBall 是一个面向领域特定架构(Domain Specific Architecture)的可扩展框架，基于 RISC-V 架构构建，专为高性能计算和机器学习加速器设计而优化。

## 项目概述

BuckyBall 框架提供了完整的硬件设计、仿真验证和软件开发工具链，支持从 RTL 设计到系统级验证的全流程开发。框架采用模块化设计，支持灵活的配置和扩展，适用于各种专用计算场景。

### 核心特性

- **可扩展架构**: 基于 Rocket-chip 和 Chipyard 框架，支持自定义 RoCC 协处理器
- **完整工具链**: 集成 Verilator、FireSim 等仿真工具，支持多种验证方法
- **统一测试框架**: 提供标准化的测试环境和工作负载管理
- **自动化工作流**: 支持 CI/CD 集成和自动化构建部署

## 快速开始

### 环境依赖

在开始之前，请确保系统满足以下依赖要求：

**必需软件**:
- Anaconda/Miniconda (Python 环境管理)
- Ninja Build System
- GTKWave (波形查看器)
- Bash Shell 环境

**安装依赖**:
```bash
# 安装 Anaconda
# 下载地址: https://www.anaconda.com/download/

# 安装系统工具
sudo apt install ninja-build gtkwave

# 可选：FireSim 免密配置
# 在 /etc/sudoers 中添加: user_name ALL=(ALL) NOPASSWD:ALL
```

### 源码构建

**1. 克隆仓库**
```bash
git clone https://github.com/DangoSys/buckyball.git
cd buckyball
```

**2. 初始化环境**
```bash
./scripts/init.sh
```
*注意: 初始化过程约需 2 小时，包括依赖下载和编译*

**3. 环境激活**
```bash
source buckyball/env.sh
```

### 验证安装

运行 Verilator 仿真测试验证安装：
```bash
bbdev verilator --run '--jobs 16 --binary ctest_mvin_mvout_alternate_test_singlecore-baremetal --batch'
```

### Docker 快速体验 (很久没更新了)

```bash
docker pull ghcr.io/dangosys/buckyball:latest
```
*注意: Docker 镜像可能不是最新版本，建议使用源码构建*

## 其他

You can learn more from [DeepWiki](https://deepwiki.com/DangoSys/buckyball) and [Zread](https://zread.ai/DangoSys/buckyball)
