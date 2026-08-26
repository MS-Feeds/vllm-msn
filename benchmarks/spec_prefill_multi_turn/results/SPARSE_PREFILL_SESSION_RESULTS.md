# Sparse-prefill scope: results recovered from session

Everything measured while evaluating `--sparse-prefill` (the opt-in scope
that restricts each turn's PREFILL to the speculator's selected blocks, on
top of the default decode-only restriction). Raw console/CSV output is
reproduced verbatim; anything computed from it is labelled **derived**.

Model: Llama-3.1-8B-Instruct target / Llama-3.2-1B-Instruct scorer.
`enforce_eager=True`, `--target-gpu-memory-utilization 0.85`,
`--speculator-gpu-memory-utilization 0.2`, `max_tokens=64`, `rep=1`.

---

## A. Decode-path regression check — `sparse_decode_microbench.py`

```
====================================================================================================
keep_rate | ms/tok(med) | ms/tok(p90) |  KV bytes/step | attn ms/step | other ms/step |  roofline ms/tok | meas/roofline
------------------------------------------------------------------------------------------------------------------------
      1.0 |      23.004 |      27.011 | 10,127,147,008 |        0.000 |         0.000 |           13.095 |          1.76
      0.2 |      24.028 |      28.043 |  2,053,111,808 |        0.000 |         0.000 |            9.058 |          2.65
====================================================================================================
```

**Verdict: pass.** KV bytes/step fell 4.93× against a 5× keep-rate change,
so the block gather is genuinely skipping blocks — this is not the old
`max_seq_len` symptom where bytes stayed flat.

Notes:
- `attn ms/step` / `other ms/step` of exactly `0.000` is a **known-broken
  instrument**, documented at `sparse_decode_microbench.py:90`:
  `run_profiled_decode` wraps the driver process while vLLM runs CUDA work
  in a separate EngineCore subprocess, so `torch.profiler` sees no kernels.
- Wall-clock is flat (k=0.2 is ~1 ms/tok *slower*). **Derived**: with
  `--weights-gib 14.96` / `--bandwidth-tbps 2.0`, weights alone are
  8.03 ms/tok of the roofline, so the entire prize from a 5× KV cut is
  4.03 ms/tok — buried under ~10 ms/step of eager-mode launch/dispatch
  overhead the roofline doesn't model. The ~1 ms regression at k=0.2 is
  consistent with the per-layer metadata-patch cost `pop_override_timing`
  already measures at 0.5–1 s/turn.
- If the run printed `[FLAT -- EVICTION NOT REDUCING KV TRAFFIC]`, its
  diagnosis was wrong here: the ms/tok trigger fired (4.5% < 5% threshold)
  while the KV-bytes trigger — which is what the message actually
  describes — did not.

---

## B. Paired smoke test — 1 conversation, `SPARSE-k80-g32`

`scbench_qa_eng`, conversation `scbench_qa_eng-0`, 5 turns, decode-only vs
`--sparse-prefill`. Turn-0 context 87,920 tokens.

### B1. FLOPs/turn summary line (sparse-prefill arm)

```
spec_prefill=136.14 spec_lookahead=0.11 spec_scoring=0.05 target_prefill=652.27 target_decode=1.72
```

### B2. Both arms, from `all_runs.csv`

```
6790deb65791e0c5aa438b6] prefill=652.4687908601857 decode=1.4966800252928 total=790.229876473856
a438b6] [prefill=sparse] prefill=652.266281893888 decode=1.7246519820288 total=790.290454675456
```

| | prefill/turn | decode/turn | total/turn |
|---|---:|---:|---:|
| decode-only | 652.4688 | 1.49668 | 790.2299 |
| sparse-prefill | 652.2663 | 1.72465 | 790.2905 |
| **Δ** | **−0.2025** | **+0.2280** | **+0.0606 (+0.0077%)** |

**Derived:** the prefill delta of −0.2025 TF/turn matches the −0.2026
predicted from the per-turn attended lengths in B3 — validating the whole
accounting path (worker measurement → `pop_prefill_steps` →
`target_sparse_prefill_flops` → CSV). The decode *increase* is not a cost
of the mechanism: the decode selection is identical between scopes, and
+0.228 TF/turn works out to exactly 21 extra decode steps (0.0542 TF/step
at attended ≈ 74,800) from turns 2–3 generating longer answers.

**The run cost 0.0077% more overall** — the prefill saving was slightly
more than cancelled by output-length variance.

### B3. Per-turn pairing

```
('scbench_qa_eng-0', 0) IDENTICAL | chunks=1 attended_mean=87920 delta=87920 resident=0
('scbench_qa_eng-0', 1) IDENTICAL | chunks=1 attended_mean=74606 delta=34 resident=87932
('scbench_qa_eng-0', 2) differs | chunks=1 attended_mean=75023 delta=32 resident=87983
('scbench_qa_eng-0', 3) differs | chunks=1 attended_mean=74817 delta=35 resident=88062
('scbench_qa_eng-0', 4) IDENTICAL | chunks=1 attended_mean=74612 delta=44 resident=88152
```

**Validation verdict: pass on every criterion.**

| check | result |
|---|---|
| turn 0 `IDENTICAL` | ✅ the critical one |
| turn 0 `attended_mean == delta`, resident 0 | ✅ dense, gather correctly degenerate |
| turns 1+ `attended_mean` < full | ✅ 74.6k vs 88.0k |
| some turns differ | ✅ turns 2, 3 — the restriction is not inert |

Effective keep 84.6–85.2% against a nominal 80% — the +5pp is block
granularity rounding up, the wrapper-span union, and the contiguous
force-kept tail.

### B4. Steady-state prefill, turn 0 excluded

```
decode-only     2.1805 TF/turn
sparse-prefill  1.9274 TF/turn
reduction       11.61%
   scbench_qa_eng-0 t1: 2.0434 -> 1.8053 TF (-11.7%)  resident drift +0
   scbench_qa_eng-0 t2: 1.9241 -> 1.7061 TF (-11.3%)  resident drift +0
   scbench_qa_eng-0 t3: 2.1057 -> 1.8622 TF (-11.6%)  resident drift +8
   scbench_qa_eng-0 t4: 2.6488 -> 2.3359 TF (-11.8%)  resident drift +21
```

**Derived, and the measurement validates itself two ways.** Turns 1–2 have
zero drift (strictly controlled) and give −11.7% / −11.3%; turns 3–4 carry
+8 / +21 drift and give −11.6% / −11.8% — the drifted turns land inside
the spread of the undrifted ones, so contamination isn't reaching the
signal (21 tokens against an 88k resident is 0.024% of the input).

It also matches theory exactly: turn 3's prefill is `2.1059 TF = linear
0.4896 (23.2%) + attn 1.6163 (76.8%)`; effective keep 84.9% ⟹ attention
falls 15.1% ⟹ `0.151 × 76.8% = 11.6%`. Measured 11.6%. No residual.

**Drift** = difference in `target_resident_len` between the two runs,
which can only come from output-length divergence (`target_resident_len
+= len(delta_ids) + len(actual_output_ids)`; the delta is dataset text and
identical in both). The +21 at t4 is the same 21 extra decode steps
derived independently in B2.

### B5. Whole-run accounting for this conversation (derived)

```
 turn | full_len | attended | eff.keep | dense TF | sparse TF | saving
    0 |    87920 |    87920 |   100.0% |  3253.62 |   3253.62 |   0.0%
    1 |    87966 |    74606 |    84.8% |     2.04 |      1.81 |  11.7%
    2 |    88015 |    75023 |    85.2% |     1.92 |      1.71 |  11.3%
    3 |    88097 |    74817 |    84.9% |     2.11 |      1.86 |  11.6%
    4 |    88196 |    74612 |    84.6% |     2.65 |      2.34 |  11.8%

run total   dense=3262.3 TF   sparse-prefill=3261.3 TF   saving=1.013 TF (0.031%)
per-turn mean  dense=652.47   sparse-prefill=652.27
turn 0 alone = 3254 TF = 99.73% of the run's total prefill
```

Reproduces the reported 652.27 exactly.

---

## C. Full sweep — all 3 configs, g32, `--sparse-prefill`

967 matched turns (966 for k60), 193 conversations, 47 skipped.

### C1. `all_runs.csv` rows (verbatim)

```
2026-08-24T18:27:23.436964+00:00,SPARSE-k80-g32,Sparse attention (persistent cache) keep=80% granularity=32 [scorer=9213176726f574b556790deb65791e0c5aa438b6] [prefill=sparse],sparse,keep,0.8,32,32,8,13,0.85,0.2,131072,1,0,64,239,193,967,47,7759.476973772049,0.12462180160706379,40.20454390555466,7.918932409296361,1.4110158854368737,0.8000867494549507,5390.861602948888,112.0610237121582,30853.2901763916,90748.88934850051,37.65873836608066,17.32413507115342,4.6930998214300255,850,117,0,215.33233798109057,0.1386819721408728,0.059455209167023786,999.5535658555788,2.331809094468567,1217.4158501124457,1177241.1270587351,0.17703932074031328,151.71655654600812,
2026-08-24T20:37:54.353704+00:00,SPARSE-k60-g32,Sparse attention (persistent cache) keep=60% granularity=32 [scorer=9213176726f574b556790deb65791e0c5aa438b6] [prefill=sparse],sparse,keep,0.6,32,32,8,13,0.85,0.2,131072,1,0,64,239,193,966,47,7793.584712505341,0.12394809777970166,40.38126794044218,7.950395963206795,1.4310948882651977,0.6000577754636781,5385.141988471922,94.19536590576172,30855.25131225586,90707.74327122154,37.821946169772254,17.168333747619137,4.687958282069545,858,108,0,215.55844555162446,0.13866358169175982,0.059446016589383026,1000.133059860815,1.9493273245861367,1217.8389423353067,1176432.4182959064,0.17716345540418885,150.94882030450506,
2026-08-24T22:47:34.072162+00:00,SPARSE-k40-g32,Sparse attention (persistent cache) keep=40% granularity=32 [scorer=9213176726f574b556790deb65791e0c5aa438b6] [prefill=sparse],sparse,keep,0.4,32,32,8,13,0.85,0.2,131072,1,0,64,239,193,967,47,7744.506673812866,0.12486269826195595,40.12697758452262,7.882784855032904,1.3631177422920246,0.40003361838732054,5360.944806111759,73.17280769348145,30843.121671676636,90749.30299896587,37.8076525336091,17.12613730703063,4.720765510296907,863,104,0,215.33913018417982,0.13870244051789038,0.05946416489523888,998.6114979852284,1.5229973103337497,1215.671792085155,1175554.622946345,0.1772989208048475,151.79206016069998,
2026-08-25T00:55:59.327417+00:00,SPARSE-k20-g32,Sparse attention (persistent cache) keep=20% granularity=32 [scorer=9213176726f574b556790deb65791e0c5aa438b6] [prefill=sparse],sparse,keep,0.2,32,32,8,13,0.85,0.2,131072,1,0,64,239,193,967,47,7669.451075077057,0.12608464289476992,39.73808847190185,7.802611280745039,1.2766572591254262,0.20000787517807678,5345.184963023157,51.64623260498047,30841.715908050537,90748.21096173734,37,16.53285830480344,4.665131787106487,890,77,0,215.32732810834068,0.13863558566519132,0.05943585090342916,998.0849112051844,1.0420654328216752,1214.6523761829153,1174568.847768879,0.17743792690893578,153.1490110923066,
```

### C2. The apparent non-monotonicity, resolved (derived)

`target_prefill_tflops_per_turn_mean` reads k60 (1000.133) **higher** than
k80 (999.554), which cannot be a real effect. Cause: k60 ran 966 turns, the
others 967, and dividing a ~1000 TF/turn mean by one fewer turn inflates it
by ~1.035 TF — about 2.3× the genuine k80→k60 difference of ~0.45 TF.

| keep | reported mean | turns | TOTAL prefill | mean normalized to 967 |
|---|---:|---:|---:|---:|
| k80 | 999.554 | 967 | 966,568.3 | 999.554 |
| k60 | **1000.133** | 966 | 966,128.5 | **999.103** |
| k40 | 998.611 | 967 | 965,657.3 | 998.611 |
| k20 | 998.085 | 967 | 965,148.1 | 998.085 |

Totals fall monotonically: −439.8, −471.2, −509.2 TF per step down in keep
rate. Nothing went up.

**Headline:** the full k80→k20 sweep moves total FLOPs/turn by **0.227%**
(1217.42 → 1214.65) and wall time by 1.2% (7759 s → 7669 s), while costing
30.2 points of scbench_kv accuracy (79.0 → 48.8).

### C3. Grades (verbatim)

`SPARSE-k20-g32-full-pf` — overall **41.01** (967/1201 matched)

| Config | Score |
|---|---:|
| scbench_kv | 48.80 |
| scbench_qa_eng | 24.06 |
| scbench_summary | 35.56 |

| Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| | 36.37 | 41.94 | 42.71 | 44.39 | 39.85 | 29.63 | 16.67 |

`SPARSE-k40-g32-full-pf` — overall **48.79** (967/1201)

| Config | Score |
|---|---:|
| scbench_kv | 63.60 |
| scbench_qa_eng | 23.57 |
| scbench_summary | 36.08 |

| Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| | 41.02 | 49.89 | 51.50 | 52.01 | 49.88 | 26.09 | 8.16 |

`SPARSE-k60-g32-full-pf` — overall **53.47** (966/1201)

| Config | Score |
|---|---:|
| scbench_kv | 72.00 |
| scbench_qa_eng | 25.56 |
| scbench_summary | 36.26 |

| Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| | 45.38 | 53.95 | 55.05 | 57.17 | 56.07 | 24.24 | 35.71 |

`SPARSE-k80-g32-full-pf` — overall **57.03** (967/1201)

| Config | Score |
|---|---:|
| scbench_kv | 79.00 |
| scbench_qa_eng | 26.43 |
| scbench_summary | 35.89 |

| Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| | 49.99 | 60.16 | 57.82 | 60.06 | 57.50 | 25.81 | 19.51 |

### C4. Against the published decode-only baselines

Decode-only figures from `README.md`'s full graded sweep (same matched
population: 500 + 117 + 350 = 967).

**scbench_kv** — 500 turns, M000 = 81.6

| keep | decode-only | sparse-prefill | Δ | steady-state prefill |
|---|---:|---:|---:|---:|
| k80 | 79.2 | 79.00 | −0.20 | −12.9% |
| k60 | 72.8 | 72.00 | −0.80 | −27.9% |
| k40 | 65.4 | 63.60 | −1.80 | −44.3% |
| k20 | 56.6 | 48.80 | **−7.80** | −62.2% |

**scbench_qa_eng** — 117 turns, M000 = 27.26

| keep | decode-only | sparse-prefill | Δ |
|---|---:|---:|---:|
| k80 | 26.63 | 26.43 | −0.20 |
| k60 | 28.57 | 25.56 | −3.01 |
| k40 | 27.20 | 23.57 | −3.63 |
| k20 | 26.25 | 24.06 | −2.19 |

**scbench_summary** — 350 turns, M000 = 36.04

| keep | decode-only | sparse-prefill | Δ |
|---|---:|---:|---:|
| k80 | 36.04 | 35.89 | −0.15 |
| k60 | 36.40 | 36.26 | −0.14 |
| k40 | 36.16 | 36.08 | −0.08 |
| k20 | 35.92 | 35.56 | −0.36 |

**All twelve deltas are negative** — two-sided sign test p ≈ 0.0005. Most
are individually inside their own noise bands; the consistency is not.
It's also the only direction physically available: restricting what a
prefill token may attend to cannot add information.

Magnitude scales on both expected axes — by keep rate on the config that
responds to it (kv: −0.20 → −7.80), and by task type (kv ≫ qa_eng ≫
summary, matching the README's own finding that summary is flat across the
entire decode-only grid).

Caveats: decode-only numbers are the published run, not a paired re-run
(sound, since decode-only is unchanged by the patch and temperature is 0);
k60-pf matched 966 vs 967.

### C5. Cost/benefit (derived)

| keep | kv accuracy cost | steady-state prefill | whole-pipeline FLOPs |
|---|---:|---:|---:|
| k80 | −0.2 | −12.9% | −0.03% |
| k60 | −0.8 | −27.9% | −0.07% |
| k40 | −1.8 | −44.3% | −0.11% |
| k20 | −7.8 | −62.2% | −0.15% |

Turn 0's cold prefill is ~99.7% of the FLOPs and is structurally
off-limits (restricting the prefill that first computes the context's KV
would poison the persistent cache every later turn's selection reads
from). The best case buys 0.15% of total compute for 7.8 points of kv
accuracy.

**Conclusion: `--sparse-prefill` should stay off by default.** It works
exactly as designed and costs measurable accuracy for a rounding error in
total cost, because the resumable session already made steady-state
prefill nearly free. It would matter in a workload where per-turn inputs
are large relative to the initial context — many turns, or turns that
paste documents — which is not SCBench's shape.

---

## D. Granularity sweep — `scbench_qa_eng`, 12 rows, `--sparse-prefill`

117 turns from 23 conversations (47 of 69 skipped for exceeding the
speculator budget — matches the README's published qa_eng set).

### D1. Timing & generation (derived from `all_runs.csv`)

| gran | keep | actual keep | turns | s/turn (excl t0) | TTFT p50 | out_len | stop/length |
|---:|---:|---:|---:|---:|---:|---:|---:|
| g16 | 80% | 0.8001 | 117 | 1.094 | 95.2 | 25.9 | 107/10 |
| g16 | 60% | 0.6000 | **116** | 1.015 | 78.7 | 25.6 | 108/8 |
| g16 | 40% | 0.4000 | 117 | 0.983 | 63.3 | 25.6 | 109/8 |
| g16 | 20% | 0.2000 | 117 | 0.964 | 45.6 | 25.7 | 111/6 |
| g32 | 80% | 0.8001 | 117 | 1.089 | 90.2 | 26.3 | 108/9 |
| g32 | 60% | 0.6001 | **116** | 1.029 | 76.5 | 26.2 | 108/8 |
| g32 | 40% | 0.4000 | 117 | 0.989 | 60.3 | 26.7 | 108/9 |
| g32 | 20% | 0.2000 | 117 | 0.942 | 43.0 | 25.4 | 114/3 |
| g64 | 80% | 0.8002 | 117 | 1.041 | 89.9 | 25.2 | 113/4 |
| g64 | 60% | 0.6001 | 117 | 1.008 | 73.6 | 25.8 | 110/7 |
| g64 | 40% | 0.4001 | 117 | 0.951 | 56.7 | 24.6 | 111/6 |
| g64 | 20% | 0.2000 | 117 | 0.919 | 41.3 | 24.2 | 112/5 |

`ttft_p90_ms` omitted — 18,107–18,133 ms in every row, i.e. turn 0's own
TTFT surfacing at the 90th percentile (23 turn-0s of 117 turns ≈ top 20%).

TTFT p50 is monotonic on **both** axes: falling with keep rate within each
granularity, and falling with coarser granularity at every keep rate. A
2.3× spread from g16-k80 to g64-k20.

### D2. FLOPs per turn, all turns (derived)

| gran | keep | spec | prefill | prefill @117 | decode | total/turn | spec share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| g16 | 80% | 162.39 | 765.85 | 765.85 | 1.459 | 929.70 | 17.5% |
| g16 | 60% | 163.78 | 772.22 | **765.62** | 1.231 | 937.22 | 17.5% |
| g16 | 40% | 162.38 | 765.38 | 765.38 | 0.984 | 928.74 | 17.5% |
| g16 | 20% | 162.38 | 765.09 | 765.09 | 0.710 | 928.18 | 17.5% |
| g32 | 80% | 162.39 | 765.80 | 765.80 | 1.438 | 929.63 | 17.5% |
| g32 | 60% | 163.78 | 772.16 | **765.56** | 1.201 | 937.13 | 17.5% |
| g32 | 40% | 162.40 | 765.32 | 765.32 | 0.972 | 928.69 | 17.5% |
| g32 | 20% | 162.37 | 765.05 | 765.05 | 0.662 | 928.09 | 17.5% |
| g64 | 80% | 162.38 | 765.78 | 765.78 | 1.353 | 929.51 | 17.5% |
| g64 | 60% | 162.38 | 765.54 | 765.54 | 1.146 | 929.07 | 17.5% |
| g64 | 40% | 162.36 | 765.29 | 765.29 | 0.858 | 928.51 | 17.5% |
| g64 | 20% | 162.38 | 765.03 | 765.03 | 0.606 | 928.01 | 17.5% |

The k60 denominator artifact **reproduced twice** (g16 and g32 ran 116
turns, g64 ran 117). Renormalized to 117, prefill is monotonic in all
three granularities. Prefill spans only 0.11% across the whole sweep —
turn 0 domination again.

### D3. Steady state, turn 0 excluded (verbatim)

```
row                       n_t0  n_1+ |   specTF prefillTF decodeTF   totalTF |  spec%    turn0 TF
SPARSE-k20-g16-qaeng-pf     23    94 |    0.901     0.860    0.715     2.475 |  36.4%      4711.5
SPARSE-k20-g32-qaeng-pf     23    94 |    0.887     0.810    0.662     2.359 |  37.6%      4711.5
SPARSE-k20-g64-qaeng-pf     23    94 |    0.900     0.778    0.610     2.289 |  39.3%      4711.4
SPARSE-k40-g16-qaeng-pf     23    94 |    0.899     1.215    0.985     3.099 |  29.0%      4711.8
SPARSE-k40-g32-qaeng-pf     23    94 |    0.920     1.146    0.951     3.017 |  30.5%      4711.9
SPARSE-k40-g64-qaeng-pf     23    94 |    0.878     1.102    0.863     2.843 |  30.9%      4711.7
SPARSE-k60-g16-qaeng-pf     23    93 |    0.905     1.528    1.254     3.687 |  24.5%      4712.0
SPARSE-k60-g32-qaeng-pf     23    93 |    0.903     1.455    1.243     3.601 |  25.1%      4711.9
SPARSE-k60-g64-qaeng-pf     23    94 |    0.904     1.413    1.146     3.462 |  26.1%      4712.0
SPARSE-k80-g16-qaeng-pf     23    94 |    0.912     1.798    1.527     4.237 |  21.5%      4712.0
SPARSE-k80-g32-qaeng-pf     23    94 |    0.915     1.743    1.495     4.153 |  22.0%      4712.0
SPARSE-k80-g64-qaeng-pf     23    94 |    0.902     1.711    1.399     4.012 |  22.5%      4712.0
```

**This is the cleanest data in the set.** Independent checks that pass:

- `turn0 TF` is 4711.4–4712.0 across all twelve — a 0.01% spread. Turn 0's
  prefill is dense under every configuration, so it must be near-constant.
  The residual is *monotonic in keep rate* (k80 → 4712.0, k20 → 4711.4),
  which is turn 0's own decode shrinking. **Derived**: that term should
  span ~1.4 TF at k80 down to ~0.63 at k20 — a 0.77 TF difference against
  an observed 0.6, on a quantity that is 0.01% of the total.
- `specTF` is 0.878–0.920 everywhere: flat, as it must be.
- Prefill and decode are monotonic on both axes with no exceptions.
- The k60 anomaly disappears — per-turn means are immune once turn 0 is out
  (1.528 sits correctly between k80's 1.798 and k40's 1.215).

**`spec%` rises 21.5% → 39.3%** as keep tightens. At k20-g64 nearly 40% of
ongoing compute is the 1B scorer deciding what to drop. `all_runs.csv`
reports a flat 17.5% for all twelve rows — that is a turn-0 statistic, not
a steady-state one.

Turn 0 is ~4712 TF against a steady turn of 2.3–4.2 — about **1150×** —
putting it at 99.6–99.8% of each run. The 1.85× steady-state range across
this table is a range over 0.2–0.4% of the work.

### D4. Granularity mechanism (derived, corrects an earlier prediction)

g64 is **cheapest** at every keep rate, not most expensive. The engine's KV
block size is 16; a 16-token scoring chunk that isn't block-aligned
straddles two KV blocks, so the gather keeps 32 tokens to get 16, while a
64-token run straddles at most five blocks instead of four. Contiguity
*reduces* block-boundary waste rather than raising effective keep.

Combined with the published decode-only grid (k20: g64 = 62.2 vs g16 =
46.6 on kv), **g64 wins on both cost and accuracy; g16 has no argument
left.**

---

## E. Still missing

- `M000` baseline with FLOP records on any config — the published
  `M000_predictions.jsonl` predates the FLOP model (no `flops` field), so
  the steady-state baseline comparison is **derived, not measured**.
  Estimate for qa_eng: ~4.3 TF/turn (~2.6 prefill + ~1.7 dense decode over
  a ~98k context), which would put `SPARSE-k80-g16` at 4.237 roughly at
  break-even and only the aggressive rows ahead.
- Grades for the 12-row qa_eng granularity sweep (D) — cost side only so far.
- `scbench_kv` and `scbench_summary` granularity sweeps.
- Paired decode-only arms with `flops` for any granularity sweep.
- Multi-chunk prefill gather never executed on hardware: every turn
  reported `chunks=1`, and SCBench deltas (30–50 tokens) can't span a
  realistic chunk size. Unit-test coverage only.

## F. Reproduction

```bash
python3 grade_scbench.py --batch --samples datasets/scbench_samples.jsonl
```

```bash
python3 compare_scopes.py --exp SPARSE-k80-g32,SPARSE-k60-g32,SPARSE-k40-g32,SPARSE-k20-g32 --a-suffix=-full-dense --b-suffix=-full-pf --reference M000
```

Steady-state FLOP readout used for D3:

```bash
python3 -c "
import json,glob,os,statistics,sys
files=[]
for pat in sys.argv[1:]:
    hits=sorted(glob.glob(pat))
    if not hits: print('  no match: %s'%pat)
    files+=hits
print('%-24s %5s %5s | %8s %9s %8s %9s | %6s %11s'%('row','n_t0','n_1+','specTF','prefillTF','decodeTF','totalTF','spec%','turn0 TF'))
for p in files:
    rs=[json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]
    if not rs or 'flops' not in rs[0]:
        print('  %s: no flops field'%os.path.basename(p)); continue
    t0=[r['flops'] for r in rs if r['turn_idx']==0]
    st=[r['flops'] for r in rs if r['turn_idx']>0]
    if not st: continue
    m=lambda k,rows: statistics.mean(f.get(k,0) for f in rows)/1e12
    spec=sum(m(k,st) for k in ('spec_prefill','spec_lookahead','spec_scoring'))
    pre,dec=m('target_prefill',st),m('target_decode',st)
    tot=spec+pre+dec
    print('%-24s %5d %5d | %8.3f %9.3f %8.3f %9.3f | %5.1f%% %11.1f'%(
        os.path.basename(p).replace('_predictions.jsonl',''),len(t0),len(st),
        spec,pre,dec,tot,100*spec/tot,m('total',t0)))
" 'results/M000*qaeng*_predictions.jsonl' 'results/SPARSE-k*-g*-qaeng-pf_predictions.jsonl'
```
