# CI 说明

push/PR 到 `main` 会自动触发 `check.yaml`：

1. **pre-commit**：拉代码, 并校验格式
2. **design-check**：通过后，matrix 并行跑各 device 的正确性测试（BEMU bare/Linux、Verilator DiffTest）；性能由 bbbench 评估（共享同一个 `~/Code/buckyball` 环境，基于 `ci_repo_lock.sh` 协调）
