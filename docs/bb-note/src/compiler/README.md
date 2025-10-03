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

注意:
1. bert, whisper, stable-diffusion, llama2, DeepseekR1 模型下载需要提前配置好 huggingface 的访问
2. whisper 暂不支持
3. llama2 的模型下载需要额外填写 API-key 或使用缓存的凭证
