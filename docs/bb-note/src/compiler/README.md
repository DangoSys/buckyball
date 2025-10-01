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
    -DMODEL="lenet" \
    -DARCH="gemmini"
ninja
```
