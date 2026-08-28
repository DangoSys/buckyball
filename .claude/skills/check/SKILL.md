---
name: check
description: Staticaly validate Buckyball Ball registration consistency and optionally auto-fix mismatches. Use this skill when users ask to inspect registration status, validate Ball configuration, troubleshoot registration issues, or verify consistency after registration edits.
---

## Validation Flow

Call MCP tool `validate(chip=..., balldomain?=...)`.

Default `chip=toy`. If `balldomain` is omitted, uses the file referenced by
`examples/chips/<chip>/configs/tiles/cores/default.toml` (`balldomain = ...`).
You can also pass a stem such as `default` / `full`.

Checks on that TOML:

1. `ballNum` equals `ballIdMappings` length
2. `ballId` is strictly increasing (`0, 1, 2, ...`) with no gaps
3. no duplicated `ballId` / `ballName`
4. no duplicated `funct7` / `mnemonic` in `ballISA` (scope = this single balldomain / core; other cores may reuse the same funct7)
5. every `ballISA.bid` exists in mappings; every ball has ≥1 ISA entry
6. relative `config=` paths exist; `inBW`/`outBW` are positive
7. ball ISA headers / MLIR must not hardcode ball `funct7` (encoding is generated from this TOML into `examples/cores/<core>/isa/ballISA.h`)

Report pass/fail for each item.

## Registration Summary

After validation, print the `balls` array from the tool result as a table:

| ballId | ballName | funct7 | mnemonic | inBW | outBW | config |
|--------|----------|--------|----------|------|-------|--------|

Data source: `examples/chips/<chip>/configs/tiles/cores/balldomains/*.toml`

## Auto Fix

If validation finds inconsistencies and they are deterministic to fix, ask whether to auto-fix:

1. **`ballNum` mismatch** — set `ballNum` to `ballIdMappings` length
2. **non-contiguous `ballId`** — renumber to `0, 1, 2, ...` and sync `ballISA.bid`
3. **missing ISA row** — add a `ballISA` entry for the orphan `ballId`
4. **broken `config=` path** — fix the relative path to the ball's config toml

For non-auto-fixable issues (for example, `funct7` conflicts), provide root-cause analysis and manual fix guidance.
