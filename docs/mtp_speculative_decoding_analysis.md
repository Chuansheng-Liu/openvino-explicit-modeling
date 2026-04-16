# MTP Speculative Decoding: Cost Analysis

## Verification Cost = Generation Cost

`run_main_step(next_id)` performs a standard KV-cache decode step:

1. Feed 1 token through all transformer layers (full forward pass)
2. Update KV cache
3. Read logits → argmax → return token ID

This is **identical** to every other decode step. The model has no cheaper "verify only" path — it simply sees a token, processes it, and produces the next predicted token. Verification happens externally: `if (ref0 == draft)`.

## Sequential MTP Cost Breakdown

```
Normal decode:   [main]                   → 1 token,  cost = T_main
MTP (reject):    [mtp] + [main]           → 1 token,  cost = T_mtp + T_main
MTP (accept):    [mtp] + [main] + [main]  → 2 tokens, cost = T_mtp + 2×T_main
```

Every step pays `T_mtp` as **pure overhead**. Accepting only saves one main model call on that step — it never eliminates the mandatory first verification call.

## Throughput Formula

Let:
- `p` = accept rate (fraction of draft tokens accepted)
- `T_main` = main model decode latency per token
- `T_mtp` = MTP draft model latency per token

```
Tokens per step = 1 + p        (accept: 2 tokens, reject: 1 token, weighted)
Time per step   = T_mtp + (1+p)·T_main

Speedup vs. greedy = [(1+p)·T_main] / [T_mtp + (1+p)·T_main]
                   = 1 / [1 + T_mtp / ((1+p)·T_main)]
                   < 1 always (since T_mtp > 0)
```

### Qwen3.5-2B example (measured)

| Parameter | Value |
|-----------|-------|
| T_main    | ~16.7 ms/token (60 tok/s greedy) |
| T_mtp     | ~3.4 ms/token (estimated) |
| p         | ~0.23 (23% accept rate) |
| Speedup   | 1.23×16.7 / (3.4 + 1.23×16.7) ≈ **0.83** |
| Predicted | 60 × 0.83 ≈ **50 tok/s** |
| Measured  | **50 tok/s** ✓ |

### Qwen3.5-35B-A3B (apparent improvement was not MTP)

The observed improvement from ~20→25 tok/s was from switching **sampling → greedy** (`temperature 0`), not from MTP itself. The 35B model is slow enough that T_mtp/T_main is small, masking the overhead, but there is still no net speedup from MTP alone.

## Why Batched Speculative Decoding Is Different

Leviathan et al. (2023) verify K draft tokens in **one** main model call by running them as a batch `[s, s+d1, s+d2, ..., s+dK]` through attention simultaneously:

```
Batched MTP:  [mtp×K] + [main×1]  → up to K+1 tokens
Verification cost per draft token: T_main / K  (instead of T_main)
```

This achieves real speedup because verification becomes K× cheaper per token. Our sequential architecture cannot do this — KV cache management is strictly sequential (each step extends the cache by exactly 1 token), so K draft tokens cannot be verified in a single main model call.

## Summary

| Architecture | Can it improve throughput? | Why |
|---|---|---|
| Sequential single-step MTP (ours) | No — always slower | T_mtp adds overhead; main call count unchanged |
| Batched speculative decoding | Yes — up to K× speedup | K draft tokens verified in 1 main call |
