---
name: check
description: 对 Buckyball Ball 注册状态做静态校验，并可选自动修复不一致。当用户要求检查注册状态、校验 Ball 配置、排查注册问题，或在修改注册文件后需要验证一致性时使用此 skill。
---

## 校验流程

调用 MCP 工具 `validate`，检查以下 6 项注册不变量：
1. ballNum == ballIdMappings 数组长度
2. ballId 严格递增（0, 1, 2, ...），不跳号
3. ballId 无重复
4. DISA.scala funct7 无重复
5. busRegister.scala case 名称与 default.json ballName 一致
6. DomainDecoder.scala BID 与 default.json ballId 一致

每项报告 pass/fail 状态。

## 注册状态概览

校验后生成一张汇总表，展示当前所有 Ball 的注册信息。数据源：

- `arch/src/main/scala/framework/balldomain/configs/default.json` — ballId, ballName, inBW, outBW
- `arch/src/main/scala/examples/toy/balldomain/DISA.scala` — funct7 值
- `arch/src/main/scala/examples/toy/balldomain/DomainDecoder.scala` — 解码行中的 BID

表格格式：

| ballId | ballName | funct7 | inBW | outBW | DISA | busReg | Decoder |
|--------|----------|--------|------|-------|------|--------|---------|
| 0      | VecBall  | 32     | 2    | 4     | ok   | ok     | ok      |
| ...    | ...      | ...    | ...  | ...   | ...  | ...    | ...     |

## 自动修复

如果校验发现不一致，且属于以下可确定性修复的类型，提示用户是否自动修复：

1. **ballNum 不匹配** — 自动将 ballNum 更新为 ballIdMappings 数组长度
2. **ballId 不连续** — 自动重编号为 0, 1, 2, ...（同步更新 DomainDecoder.scala 中的 BID）
3. **busRegister.scala 缺少 case** — 提示缺少哪些 Ball，给出需要添加的 import 和 match case 代码
4. **DomainDecoder.scala BID 不一致** — 自动更新 BID 值与 default.json 匹配

对于无法自动修复的问题（如 funct7 冲突），给出原因分析和手动修复建议。
