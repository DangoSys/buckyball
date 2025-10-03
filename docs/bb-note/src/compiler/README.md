# Compiler Build Guide

## Basic Workload Compilation

To build the workload, follow these steps:

```bash
mkdir build && cd build
cmake -G Ninja ..
ninja
```

## Model-Level Testing

To enable model-level testing with specific models and architectures:

```bash
mkdir build && cd build
cmake -G Ninja .. \
    -DMODEL="lenet,resnet18,mobilenetv3,bert,stablediffusion" \
    -DARCH="gemmini,buckyball"
ninja
```

注意:
bert, stable-diffusion, llama2, DeepseekR1 模型下载需要提前配置好 huggingface 的访问
