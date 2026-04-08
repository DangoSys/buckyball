---
name: check
description: Staticaly validate Buckyball Ball registration consistency and optionally auto-fix mismatches. Use this skill when users ask to inspect registration status, validate Ball configuration, troubleshoot registration issues, or verify consistency after registration edits.
---

## Validation Flow

Call MCP tool `validate` to check these 6 invariants:
1. `ballNum` equals `ballIdMappings` array length
2. `ballId` is strictly increasing (`0, 1, 2, ...`) with no gaps
3. no duplicated `ballId`
4. no duplicated `funct7` in `DISA.scala`
5. case names in `busRegister.scala` match `ballName` in `default.json`
6. BID values in `DomainDecoder.scala` match `ballId` in `default.json`

Report pass/fail for each item.

## Registration Summary

After validation, generate a summary table for all Ball registrations. Data sources:

- `arch/src/main/scala/framework/balldomain/configs/default.json` — `ballId`, `ballName`, `inBW`, `outBW`
- `arch/src/main/scala/examples/toy/balldomain/DISA.scala` — `funct7` values
- `arch/src/main/scala/examples/toy/balldomain/DomainDecoder.scala` — BID in decode rows

Table format:

| ballId | ballName | funct7 | inBW | outBW | DISA | busReg | Decoder |
|--------|----------|--------|------|-------|------|--------|---------|
| 0      | VecBall  | 32     | 2    | 4     | ok   | ok     | ok      |
| ...    | ...      | ...    | ...  | ...   | ...  | ...    | ...     |

## Auto Fix

If validation finds inconsistencies and they are deterministic to fix, ask whether to auto-fix:

1. **`ballNum` mismatch** — update `ballNum` to `ballIdMappings` length
2. **non-contiguous `ballId`** — renumber to `0, 1, 2, ...` (and sync BID in `DomainDecoder.scala`)
3. **missing cases in `busRegister.scala`** — list missing Balls and provide required imports and `match case` entries
4. **BID mismatch in `DomainDecoder.scala`** — update BID values to match `default.json`

For non-auto-fixable issues (for example, `funct7` conflicts), provide root-cause analysis and manual fix guidance.
