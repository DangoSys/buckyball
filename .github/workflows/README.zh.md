# CI 说明

push/PR 到 `main` 会自动触发 `check.yaml`：

1. **pre-commit**：拉代码, 并校验格式
2. **chip-check**：通过后，matrix 并行跑各 chip（共享同一个 `~/Code/buckyball` 环境，基于 `ci_repo_lock.sh` 协调）

`regression.yml` 是 reviewer 的可选测试。注意：Environment `regression` 必须在仓库 Settings → Environments 里打开 Required reviewers，否则会直接跑。
