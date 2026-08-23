# CI 说明

push/PR 到 `main` 只触发 `ci.yml`：

1. **pre-commit**：拉代码 + 校验一次
2. **chip-check**：通过后，matrix 并行跑各 chip（共享 `~/Code/buckyball`，`ci_repo_lock.sh` 协调）

手动只跑 pre-commit：Actions 里触发 `pre-commit.yml`。
