# CI 说明

push/PR 到 `main` 只触发 `check.yaml`：

1. **pre-commit**：拉代码, 并校验格式
2. **chip-check**：通过后，matrix 并行跑各 chip（共享同一个 `~/Code/buckyball` 环境，基于 `ci_repo_lock.sh` 协调）

`regression.yml` 是 reviewer 的可选测试，给 PR 打上 `regression` 标签后才会真正跑这个ci。
