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
    -DMODEL="lenet,resnet18,mobilenetv3,bert,stablediffusion,llama2,deepseekr1" \
    -DARCH="gemmini,buckyball"
ninja
```

Note:
1. Model downloads for bert, whisper, stable-diffusion, llama2, DeepseekR1 require pre-configured HuggingFace access
2. whisper is currently not supported
3. llama2 model download requires additional API-key or cached credentials
