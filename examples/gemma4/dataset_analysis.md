# Dataset analysis — `layer1_delta_20260501.txt`

Source: `/nvmedata/data/layer1_delta_20260501.txt` (24.87 GB)

Rows parsed: **859,988**  (skipped format errors: 0)

Messages-per-row histogram: {2: 859988}

Roles seen: {'system': 859988, 'user': 859988}

Phase A (full stream, char-only): 184.7s
Phase B (sample tokenization, n=10,000): 152.3s


## Char-length distribution (full 859,988-row pass)

| metric | min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| system content chars/row | 3,345 | 3,345 | 3,345 | 3,345 | 3,345 | 3,345 | 3,345 | 3,345 | 3,345 | 3,345.0 |
| user content chars/row | 265 | 909 | 2,827 | 9,917 | 28,315 | 59,464 | 87,491 | 168,629 | 1,103,343 | 22,817.6 |
| total content chars/row | 3,610 | 4,254 | 6,172 | 13,262 | 31,660 | 62,809 | 90,836 | 171,974 | 1,106,688 | 26,162.6 |

## Token-length distribution (sample n=10,000)

Computed with the actual Gemma 4 tokenizer + chat template (`apply_chat_template(..., add_generation_prompt=True)` then `encode(add_special_tokens=False)`).

| metric | min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tokens per formatted prompt | 970 | 1,152 | 1,669 | 3,479 | 8,467 | 16,650 | 24,352 | 46,703 | 117,043 | 6,958.8 |
| formatted-string chars | 3,704 | 4,386 | 6,345 | 13,278 | 32,195 | 63,203 | 92,904 | 176,923 | 404,454 | 26,402.9 |

## Chars → tokens conversion factor (per-row, sample)

| min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.63 | 3.68 | 3.76 | 3.82 | 3.87 | 3.92 | 3.96 | 4.03 | 4.37 | 3.81 |

Use median chars/token ratio to estimate token counts in the full dataset's char-distribution table above when you need token-level numbers without running the tokenizer.
