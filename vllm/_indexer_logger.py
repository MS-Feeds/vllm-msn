"""Per-layer Top-K logit recorder for DSA experiments.

Originally designed for GLM-5.1 sparse attention, now extended to DeepSeek MoE.

Captures four streams per (layer, phase) per worker process:
  A. Relative-gap histogram at the top-k boundary  (decision driver)
  B. Raw-logit histogram, masked to valid (q,k) positions  (range/saturation)
  C. Moments of the top-k winners  (winner dynamic range)
  D. Radix boundary-bucket size histograms (§14 of INDEXER_LOGIT_EXPERIMENT.md)
     For each d in INDEXER_RADIX_BITS_SWEEP (default "4,6,8,12,16"), partition
     each row's valid logits by the top-d bits of their bit-flipped uint32
     representation and record the size of the bucket that contains rank k.
     Used to answer: "how many passes does MSD radix-select need for this
     workload, and does it depend on layer/phase?"

Usage modes:
  1. GLM-5.1 sparse attention: raw attention logits (INDEXER_IS_EXPERT_ROUTING=0)
  2. DeepSeek MoE: post-softmax expert scores (INDEXER_IS_EXPERT_ROUTING=1)

No-op when INDEXER_LOGIT_DUMP_DIR is unset. See INDEXER_LOGIT_EXPERIMENT.md
for the full design.
"""
from __future__ import annotations

import atexit
import os
import re
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def _env_range(name: str, default: tuple[float, float]) -> tuple[float, float]:
    v = os.environ.get(name)
    if not v:
        return default
    lo, hi = v.split(",")
    return float(lo), float(hi)


def _env_int_list(name: str, default: list[int]) -> list[int]:
    v = os.environ.get(name)
    if not v:
        return default
    return [int(x.strip()) for x in v.split(",") if x.strip()]


@dataclass
class _Config:
    dump_dir: Path
    range_lo: float
    range_hi: float
    bins: int
    gap_window: int
    gap_range_hi: float
    gap_bins: int
    eps_norm: float
    sample_k: int
    dump_every: int
    # Stream D — radix boundary-bucket sweep
    radix_bits_sweep: list[int]   # d values to probe, e.g. [4, 6, 8, 12, 16]
    radix_bkt_bins: int           # histogram bins for boundary-bucket sizes

    @classmethod
    def from_env(cls):
        dump = os.environ.get("INDEXER_LOGIT_DUMP_DIR")
        if not dump:
            return None

        # Auto-detect mode: expert routing vs attention
        # Expert routing (DeepSeek MoE): post-softmax scores in [0, 1]
        # Attention (GLM-5.1): raw logits, typically [-50, 50]
        is_expert_routing = os.environ.get("INDEXER_IS_EXPERT_ROUTING", "0") == "1"

        if is_expert_routing:
            # DeepSeek expert routing: post-softmax probabilities
            default_range = (0.0, 1.0)
        else:
            # GLM-5.1 attention: raw attention logits
            default_range = (-50.0, 50.0)

        lo, hi = _env_range("INDEXER_LOGIT_RANGE", default_range)
        return cls(
            dump_dir=Path(dump),
            range_lo=lo,
            range_hi=hi,
            bins=_env_int("INDEXER_LOGIT_BINS", 4096),
            gap_window=_env_int("INDEXER_GAP_WINDOW", 8),
            gap_range_hi=_env_float("INDEXER_GAP_RANGE_HI", 4.0),
            gap_bins=_env_int("INDEXER_GAP_BINS", 4096),
            eps_norm=_env_float("INDEXER_GAP_EPS_NORM", 1e-6),
            sample_k=_env_int("INDEXER_LOGIT_SAMPLE_K", 0),
            dump_every=_env_int("INDEXER_LOGIT_DUMP_EVERY", 0),
            radix_bits_sweep=_env_int_list("INDEXER_RADIX_BITS_SWEEP", [4, 6, 8, 12, 16]),
            radix_bkt_bins=_env_int("INDEXER_RADIX_BKT_BINS", 256),
        )


def _detect_rank() -> int:
    """Detect the worker's rank lazily, after torch.distributed has been set up.

    vLLM worker subprocesses do NOT set RANK / LOCAL_RANK in os.environ, so we
    can't rely on env. By the time the recorder is first invoked (inside the
    indexer custom op during a real forward pass), torch.distributed has been
    initialized and we can ask it. Falls back to env, then to PID.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    for var in ("RANK", "LOCAL_RANK", "VLLM_WORKER_ID"):
        v = os.environ.get(var)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    # Last resort: use the PID so per-process files don't collide.
    return os.getpid()


_CONFIG = _Config.from_env()
_ENABLED = _CONFIG is not None


def is_enabled() -> bool:
    return _ENABLED


# ---------------------------------------------------------------------------
# Reservoir (approximate, batched)
# ---------------------------------------------------------------------------


class _Reservoir:
    """Approximate batched reservoir sampler.

    Each new batch of size n updates the reservoir so that, in the limit, every
    seen value has equal probability of being retained. The approximation:
    while the buffer is not full, we copy the leading values from the batch;
    once full, each existing slot is replaced with probability ~ n / total_seen
    by a uniform random value drawn from the new batch.
    """

    def __init__(self, k: int):
        self.k = k
        self.buf = np.empty(k, dtype=np.float32)
        self.fill = 0
        self.rng = np.random.default_rng()

    def add_batch(self, values: torch.Tensor, total_seen_before: int) -> None:
        n = int(values.numel())
        if n == 0:
            return
        # Bring the batch to host once.
        host = values.detach().to(torch.float32).cpu().numpy()
        # Fill phase
        if self.fill < self.k:
            take = min(self.k - self.fill, n)
            self.buf[self.fill : self.fill + take] = host[:take]
            self.fill += take
            host = host[take:]
            total_seen_before += take
            n = host.size
            if n == 0:
                return
        # Replacement phase: approximate uniform reservoir.
        total_after = total_seen_before + n
        p_replace = n / total_after
        m = int(self.rng.binomial(self.k, p_replace))
        if m == 0:
            return
        slots = self.rng.choice(self.k, size=m, replace=False)
        picks = self.rng.integers(0, n, size=m)
        self.buf[slots] = host[picks]

    def array(self) -> np.ndarray:
        out = np.full(self.k, np.nan, dtype=np.float32)
        out[: self.fill] = self.buf[: self.fill]
        return out


# ---------------------------------------------------------------------------
# Per-stream accumulators
# ---------------------------------------------------------------------------


@dataclass
class _StreamA:
    hist: np.ndarray
    above: int = 0
    count: int = 0
    skipped: int = 0
    sum_: float = 0.0
    sum_sq: float = 0.0
    min_: float = float("inf")
    max_: float = float("-inf")
    reservoir: _Reservoir | None = None


@dataclass
class _StreamB:
    hist: np.ndarray
    below: int = 0
    above: int = 0
    count: int = 0
    sum_: float = 0.0
    sum_sq: float = 0.0
    min_: float = float("inf")
    max_: float = float("-inf")
    reservoir: _Reservoir | None = None


@dataclass
class _StreamC:
    count: int = 0
    sum_: float = 0.0
    sum_sq: float = 0.0
    min_: float = float("inf")
    max_: float = float("-inf")
    reservoir: _Reservoir | None = None


@dataclass
class _StreamD:
    """Stream D — radix boundary-bucket sizes.

    For each d in cfg.radix_bits_sweep we store, per layer/phase:
      bkt_hist[d_idx]:  histogram of per-row boundary-bucket sizes
                        (bins = 0..cfg.radix_bkt_bins-1, last bin catches >=max)
      bkt_zero[d_idx]:  rows where boundary-bucket size == 0 (clean partition)
      bkt_count[d_idx]: total rows contributing to this d
      bkt_sum[d_idx]:   sum of bucket sizes (for mean)
      bkt_sum_sq[d_idx]:sum of square bucket sizes (for variance)
      bkt_max[d_idx]:   maximum bucket size seen
    """
    # indexed by position in cfg.radix_bits_sweep
    bkt_hist: list[np.ndarray]  # each shape [radix_bkt_bins]
    bkt_zero: list[int]
    bkt_count: list[int]
    bkt_sum: list[int]
    bkt_sum_sq: list[int]
    bkt_max: list[int]


def _new_streams(cfg: _Config):
    sA = _StreamA(hist=np.zeros(cfg.gap_bins, dtype=np.int64))
    sB = _StreamB(hist=np.zeros(cfg.bins, dtype=np.int64))
    sC = _StreamC()
    nd = len(cfg.radix_bits_sweep)
    sD = _StreamD(
        bkt_hist=[np.zeros(cfg.radix_bkt_bins, dtype=np.int64) for _ in range(nd)],
        bkt_zero=[0] * nd,
        bkt_count=[0] * nd,
        bkt_sum=[0] * nd,
        bkt_sum_sq=[0] * nd,
        bkt_max=[0] * nd,
    )
    if cfg.sample_k > 0:
        sA.reservoir = _Reservoir(cfg.sample_k)
        sB.reservoir = _Reservoir(cfg.sample_k)
        sC.reservoir = _Reservoir(cfg.sample_k)
    return sA, sB, sC, sD


# ---------------------------------------------------------------------------
# Bit-flip transform: reinterpret float32 as uint32 so unsigned sort order
# matches float order (handles negatives via the standard key transform).
# ---------------------------------------------------------------------------

def _float32_to_sortable_uint32(t: torch.Tensor) -> torch.Tensor:
    """Bit-flip FP32 tensor to sortable uint32 (standard radix-sort trick).

    Positive floats: flip the sign bit → they sort above transformed negatives.
    Negative floats: flip all bits → more-negative gets smaller uint.
    Works correctly for ±0, ±inf, NaN (NaN → very large uint, sorts last).
    """
    bits = t.view(torch.int32).view(torch.uint8)  # reinterpret, no copy
    # Operate as int32 for the XOR (torch has no uint32 bitwise on older builds)
    i32 = t.view(torch.int32)
    # where sign bit set (negative float): flip all 32 bits
    # where sign bit clear (non-negative): flip only sign bit (0x80000000)
    flip_all = i32 ^ (-1)                           # ^= 0xFFFFFFFF
    flip_sign = i32 ^ torch.tensor(0x80000000, dtype=torch.int32, device=t.device)
    neg_mask = (i32 < 0)
    sortable_i32 = torch.where(neg_mask, flip_all, flip_sign)
    # Treat as unsigned: view as int32 is fine for >> shift (logical on positive
    # values coming from flip_sign; arithmetic is safe for flip_all too because
    # the result is always non-negative after the transform for non-NaN values).
    return sortable_i32


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_parse_warned = False


def _parse_layer(prefix: str):
    m = _LAYER_RE.search(prefix)
    if m is None:
        global _parse_warned
        if not _parse_warned:
            print(
                f"[_indexer_logger] could not parse layer from '{prefix}'; "
                "records from this prefix will be dropped.",
                file=sys.stderr,
            )
            _parse_warned = True
        return None
    return int(m.group(1))


class _IndexerRecorder:
    def __init__(self, cfg: _Config):
        self.cfg = cfg
        self.state: dict[tuple[int, str], tuple[_StreamA, _StreamB, _StreamC]] = {}
        self.lock = threading.Lock()
        self.rank = _detect_rank()
        self.dump_path = cfg.dump_dir / f"indexer_logits_rank{self.rank}.npz"
        self.call_count = 0
        self.seen_k = 0
        cfg.dump_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[_indexer_logger] rank={self.rank} dump_path={self.dump_path}",
            file=sys.stderr,
        )

        atexit.register(self._safe_dump)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)

                def _handler(signum, frame, _prev=prev, _self=self):
                    _self._safe_dump()
                    if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                        try:
                            _prev(signum, frame)
                        except Exception:
                            pass

                signal.signal(sig, _handler)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def record(
        self,
        k_cache_prefix: str,
        phase: str,
        logits: torch.Tensor,
        index_topk: int,
        key_valid_starts: torch.Tensor,
        key_valid_ends: torch.Tensor,
    ) -> None:
        layer = _parse_layer(k_cache_prefix)
        if layer is None:
            return
        if self.seen_k == 0:
            self.seen_k = int(index_topk)
        try:
            self._record_inner(
                layer, phase, logits, int(index_topk),
                key_valid_starts, key_valid_ends,
            )
        except Exception as e:
            print(
                f"[_indexer_logger] record() failed (layer={layer} phase={phase}): {e!r}",
                file=sys.stderr,
            )
        self.call_count += 1
        if self.cfg.dump_every > 0 and self.call_count % self.cfg.dump_every == 0:
            self._safe_dump()

    def _record_inner(self, layer, phase, logits, k, ks_t, ke_t):
        cfg = self.cfg
        nd = len(cfg.radix_bits_sweep)
        with torch.no_grad():
            logits = logits.detach()
            if logits.dtype not in (torch.float32, torch.float64):
                logits = logits.float()
            N_q, N_k = int(logits.shape[0]), int(logits.shape[1])
            w = cfg.gap_window
            need = k + w + 1

            device = logits.device
            ks = ks_t.to(device=device, dtype=torch.long).reshape(-1)
            ke = ke_t.to(device=device, dtype=torch.long).reshape(-1)
            if ks.shape[0] != N_q or ke.shape[0] != N_q:
                raise ValueError(
                    f"row count mismatch: logits {N_q}, ks {ks.shape[0]}, ke {ke.shape[0]}"
                )

            col = torch.arange(N_k, device=device).unsqueeze(0)
            valid_mask = (col >= ks.unsqueeze(1)) & (col < ke.unsqueeze(1))
            valid_per_row = (ke - ks).clamp_min(0)

            # ---- Stream B (valid raw logits) ----
            valid_vals = logits[valid_mask]
            cntB = int(valid_vals.numel())
            if cntB == 0:
                # nothing valid this call — record skip and bail
                with self.lock:
                    sA, _, _ = self._get(layer, phase)
                    sA.skipped += N_q
                return

            histB_gpu = torch.histc(
                valid_vals, bins=cfg.bins,
                min=float(cfg.range_lo), max=float(cfg.range_hi),
            )
            belowB = (valid_vals < cfg.range_lo).sum()
            aboveB = (valid_vals >= cfg.range_hi).sum()
            mnB = valid_vals.amin()
            mxB = valid_vals.amax()
            sumB = valid_vals.sum().to(torch.float64)
            sumsqB = (valid_vals.float() * valid_vals.float()).sum().to(torch.float64)

            # ---- Streams A and C: only on rows with enough valid keys ----
            rows_ok = valid_per_row >= need
            n_rows_ok = int(rows_ok.sum().item())

            histA_gpu = torch.zeros(cfg.gap_bins, dtype=torch.float32, device=device)
            aboveA = torch.tensor(0, dtype=torch.int64, device=device)
            mnA = torch.tensor(float("inf"), dtype=torch.float32, device=device)
            mxA = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
            sumA = torch.tensor(0.0, dtype=torch.float64, device=device)
            sumsqA = torch.tensor(0.0, dtype=torch.float64, device=device)
            cntA = 0

            mnC = torch.tensor(float("inf"), dtype=torch.float32, device=device)
            mxC = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
            sumC = torch.tensor(0.0, dtype=torch.float64, device=device)
            sumsqC = torch.tensor(0.0, dtype=torch.float64, device=device)
            cntC = 0

            rel_for_reservoir: torch.Tensor | None = None
            winners_for_reservoir: torch.Tensor | None = None
            skipped_A = N_q

            if n_rows_ok > 0:
                masked = torch.where(
                    valid_mask, logits,
                    torch.full_like(logits, float("-inf")),
                )
                top = torch.topk(
                    masked, need, dim=-1, largest=True, sorted=True
                ).values
                top_ok = top[rows_ok]                                    # [N_ok, need]

                # Stream C: winners (top-k values themselves)
                winners = top_ok[:, :k]                                  # [N_ok, k]
                sumC = winners.sum().to(torch.float64)
                sumsqC = (winners.float() * winners.float()).sum().to(torch.float64)
                mnC = winners.amin()
                mxC = winners.amax()
                cntC = int(winners.numel())
                winners_for_reservoir = winners.reshape(-1)

                # Stream A: boundary gap
                diffs = top_ok[:, :-1] - top_ok[:, 1:]                   # [N_ok, k+w]
                window = diffs[:, k - w : k + w]                         # [N_ok, 2w]
                window_gap = window.min(dim=-1).values                   # [N_ok]
                denom = top_ok[:, k].abs()                               # [N_ok]
                finite = denom > cfg.eps_norm
                rel = (window_gap / denom)[finite]
                cntA = int(rel.numel())
                if cntA > 0:
                    histA_gpu = torch.histc(
                        rel, bins=cfg.gap_bins,
                        min=0.0, max=float(cfg.gap_range_hi),
                    )
                    aboveA = (rel >= cfg.gap_range_hi).sum()
                    mnA = rel.amin()
                    mxA = rel.amax()
                    sumA = rel.sum().to(torch.float64)
                    sumsqA = (rel.float() * rel.float()).sum().to(torch.float64)
                skipped_A = N_q - cntA
                rel_for_reservoir = rel

            # ---- Stream D: radix boundary-bucket sizes ----
            # We need rows that have at least k valid keys (boundary exists).
            # rows_ok is already computed above (valid_per_row >= need); we
            # reuse it here — any row with >= k+w+1 valid keys is acceptable,
            # but for the radix question we only need >= k valid keys.
            rows_k_ok = valid_per_row >= k
            d_bkt_hists_np  = []
            d_bkt_zero      = []
            d_bkt_count     = []
            d_bkt_sum       = []
            d_bkt_sum_sq    = []
            d_bkt_max       = []
            n_rows_k_ok = int(rows_k_ok.sum().item())

            if n_rows_k_ok > 0:
                # Mask logits to valid positions only; replace invalid with -inf
                # so they sort below everything and never end up in the boundary
                # bucket for any row.
                masked_for_radix = torch.where(
                    valid_mask,
                    logits,
                    torch.full_like(logits, float("-inf")),
                )
                # Convert to sortable uint32 (higher = larger float value)
                sortable = _float32_to_sortable_uint32(masked_for_radix)  # [N_q, N_k]
                # -inf maps to the smallest uint32 (0x00000000 after bit-flip),
                # so invalid positions are always in the lowest bucket. Good.

                for d in cfg.radix_bits_sweep:
                    shift = 32 - d
                    # Top-d bits for each value: bucket id per (row, col)
                    bucket_ids = sortable >> shift          # [N_q, N_k], values 0..2^d-1
                    # We only process rows with >= k valid keys
                    bids_ok = bucket_ids[rows_k_ok]        # [N_rows_k_ok, N_k]
                    vmask_ok = valid_mask[rows_k_ok]        # [N_rows_k_ok, N_k]

                    # For each row, find the bucket that contains exactly rank k
                    # (0-based: the k-th largest valid value).
                    # Strategy: count, per row, how many valid values fall in
                    # each bucket descending from the top, until cumsum >= k.
                    # That bucket is the boundary bucket. Its count is the
                    # boundary-bucket size.
                    #
                    # Efficient GPU path: for each row, do a scatter-add of
                    # valid counts per bucket, then a cumulative sum from the
                    # top bucket down.
                    n_buckets = 1 << d                     # 2^d
                    Nr = bids_ok.shape[0]

                    counts = torch.zeros(
                        Nr, n_buckets, dtype=torch.int64, device=device
                    )
                    valid_flat_mask = vmask_ok.reshape(-1)
                    flat_bids = bids_ok.reshape(-1)
                    row_idx = (
                        torch.arange(Nr, device=device)
                        .unsqueeze(1)
                        .expand(Nr, N_k)
                        .reshape(-1)
                    )
                    valid_rows  = row_idx[valid_flat_mask]    # [n_valid]
                    valid_bcols = flat_bids[valid_flat_mask]  # [n_valid]
                    ones1 = torch.ones(
                        int(valid_flat_mask.sum()),
                        dtype=torch.int64, device=device,
                    )
                    counts.index_put_(
                        (valid_rows, valid_bcols),
                        ones1,
                        accumulate=True,
                    )

                    # Cumsum from the highest bucket down
                    # cumsum_from_top[r, b] = total valid keys in buckets >= b
                    cum = counts.flip(1).cumsum(dim=1).flip(1)   # [Nr, n_buckets]

                    # Boundary bucket for row r: smallest b such that
                    # cum[r, b] >= k  (i.e. the top-b buckets contain >= k keys)
                    k_tensor = torch.full(
                        (Nr, 1), k, dtype=torch.int64, device=device
                    )
                    # First bucket (from top, i.e. highest value) where
                    # cumsum crosses k: index of the last bucket with cum >= k
                    # after cum drops below k — but we want the bucket right at
                    # the crossing.  Equivalently: the bucket where
                    # cum[r,b] >= k and (b==n_buckets-1 or cum[r,b+1] < k).
                    crosses = cum >= k_tensor                        # [Nr, n_buckets]
                    # boundary bucket index (0 = highest value bucket)
                    # = last True in the crosses row (torch has no last_true;
                    #   use: n_buckets-1 - argmax(crosses.flip(1)))
                    bkt_idx = (n_buckets - 1) - crosses.flip(1).long().argmax(dim=1)
                    # size of that bucket for each row
                    bkt_size = counts.gather(1, bkt_idx.unsqueeze(1)).squeeze(1)  # [Nr]

                    bkt_size_np = bkt_size.cpu().numpy().astype(np.int64)
                    hist_bins = cfg.radix_bkt_bins
                    bkt_hist_np = np.bincount(
                        np.clip(bkt_size_np, 0, hist_bins - 1),
                        minlength=hist_bins,
                    ).astype(np.int64)
                    d_bkt_hists_np.append(bkt_hist_np)
                    d_bkt_zero.append(int((bkt_size_np == 0).sum()))
                    d_bkt_count.append(Nr)
                    d_bkt_sum.append(int(bkt_size_np.sum()))
                    d_bkt_sum_sq.append(int((bkt_size_np.astype(np.int64) ** 2).sum()))
                    d_bkt_max.append(int(bkt_size_np.max()) if Nr > 0 else 0)
            else:
                for _ in cfg.radix_bits_sweep:
                    d_bkt_hists_np.append(np.zeros(cfg.radix_bkt_bins, dtype=np.int64))
                    d_bkt_zero.append(0)
                    d_bkt_count.append(0)
                    d_bkt_sum.append(0)
                    d_bkt_sum_sq.append(0)
                    d_bkt_max.append(0)

            # ---- Single H2D for everything ----
            histA_np = histA_gpu.detach().to(torch.int64).cpu().numpy()
            histB_np = histB_gpu.detach().to(torch.int64).cpu().numpy()
            scalars = torch.stack([
                aboveA.to(torch.float64),    # 0
                sumA,                         # 1
                sumsqA,                       # 2
                mnA.to(torch.float64),       # 3
                mxA.to(torch.float64),       # 4
                belowB.to(torch.float64),    # 5
                aboveB.to(torch.float64),    # 6
                sumB,                         # 7
                sumsqB,                       # 8
                mnB.to(torch.float64),       # 9
                mxB.to(torch.float64),       # 10
                sumC,                         # 11
                sumsqC,                       # 12
                mnC.to(torch.float64),       # 13
                mxC.to(torch.float64),       # 14
            ]).cpu().numpy()

            # Reservoir source tensors: keep references so we can sample after
            # releasing the GPU work above. They are on GPU and will be moved
            # to host inside add_batch().
            res_vals_B = valid_vals if cfg.sample_k > 0 else None
            res_vals_A = rel_for_reservoir if cfg.sample_k > 0 else None
            res_vals_C = winners_for_reservoir if cfg.sample_k > 0 else None

        # ---- Commit under lock ----
        with self.lock:
            sA, sB, sC, sD = self._get(layer, phase)

            sA.hist += histA_np
            sA.above += int(scalars[0])
            sA.count += cntA
            sA.skipped += skipped_A
            sA.sum_ += float(scalars[1])
            sA.sum_sq += float(scalars[2])
            if cntA > 0:
                sA.min_ = min(sA.min_, float(scalars[3]))
                sA.max_ = max(sA.max_, float(scalars[4]))

            sB.hist += histB_np
            sB.below += int(scalars[5])
            sB.above += int(scalars[6])
            sB.sum_ += float(scalars[7])
            sB.sum_sq += float(scalars[8])
            sB.min_ = min(sB.min_, float(scalars[9]))
            sB.max_ = max(sB.max_, float(scalars[10]))
            sB.count += cntB

            sC.count += cntC
            sC.sum_ += float(scalars[11])
            sC.sum_sq += float(scalars[12])
            if cntC > 0:
                sC.min_ = min(sC.min_, float(scalars[13]))
                sC.max_ = max(sC.max_, float(scalars[14]))

            if self.cfg.sample_k > 0:
                if sA.reservoir is not None and res_vals_A is not None:
                    sA.reservoir.add_batch(res_vals_A, sA.count - cntA)
                if sB.reservoir is not None and res_vals_B is not None:
                    sB.reservoir.add_batch(res_vals_B, sB.count - cntB)
                if sC.reservoir is not None and res_vals_C is not None:
                    sC.reservoir.add_batch(res_vals_C, sC.count - cntC)

            # Stream D
            for di in range(nd):
                sD.bkt_hist[di] += d_bkt_hists_np[di]
                sD.bkt_zero[di]  += d_bkt_zero[di]
                sD.bkt_count[di] += d_bkt_count[di]
                sD.bkt_sum[di]   += d_bkt_sum[di]
                sD.bkt_sum_sq[di]+= d_bkt_sum_sq[di]
                if d_bkt_max[di] > sD.bkt_max[di]:
                    sD.bkt_max[di] = d_bkt_max[di]

    def _get(self, layer: int, phase: str):
        key = (layer, phase)
        s = self.state.get(key)
        if s is None:
            s = _new_streams(self.cfg)
            self.state[key] = s
        return s

    # ------------------------------------------------------------------
    # Dump
    # ------------------------------------------------------------------

    def _safe_dump(self) -> None:
        try:
            self.dump()
        except Exception as e:
            print(f"[_indexer_logger] dump failed: {e!r}", file=sys.stderr)

    def dump(self) -> None:
        with self.lock:
            if not self.state:
                return
            cfg = self.cfg
            layer_ids = sorted({l for l, _ in self.state.keys()})
            L = len(layer_ids)
            idx = {l: i for i, l in enumerate(layer_ids)}

            def _layer_arrays(bins):
                return dict(
                    hist_p=np.zeros((L, bins), dtype=np.int64),
                    hist_d=np.zeros((L, bins), dtype=np.int64),
                )

            A = _layer_arrays(cfg.gap_bins)
            B = _layer_arrays(cfg.bins)

            def zeros_L(dtype=np.int64): return np.zeros(L, dtype=dtype)
            def full_L(v, dtype=np.float64): return np.full(L, v, dtype=dtype)

            # Stream A scalars
            gap_above_p = zeros_L(); gap_above_d = zeros_L()
            gap_count_p = zeros_L(); gap_count_d = zeros_L()
            gap_skipped_p = zeros_L(); gap_skipped_d = zeros_L()
            gap_sum_p = zeros_L(np.float64); gap_sum_d = zeros_L(np.float64)
            gap_sum_sq_p = zeros_L(np.float64); gap_sum_sq_d = zeros_L(np.float64)
            gap_min_p = full_L(np.inf); gap_min_d = full_L(np.inf)
            gap_max_p = full_L(-np.inf); gap_max_d = full_L(-np.inf)
            # Stream B scalars
            count_p = zeros_L(); count_d = zeros_L()
            below_p = zeros_L(); below_d = zeros_L()
            above_p = zeros_L(); above_d = zeros_L()
            sum_p = zeros_L(np.float64); sum_d = zeros_L(np.float64)
            sum_sq_p = zeros_L(np.float64); sum_sq_d = zeros_L(np.float64)
            min_p = full_L(np.inf); min_d = full_L(np.inf)
            max_p = full_L(-np.inf); max_d = full_L(-np.inf)
            # Stream C scalars
            tk_count_p = zeros_L(); tk_count_d = zeros_L()
            tk_sum_p = zeros_L(np.float64); tk_sum_d = zeros_L(np.float64)
            tk_sum_sq_p = zeros_L(np.float64); tk_sum_sq_d = zeros_L(np.float64)
            tk_min_p = full_L(np.inf); tk_min_d = full_L(np.inf)
            tk_max_p = full_L(-np.inf); tk_max_d = full_L(-np.inf)
            # Stream D — radix boundary-bucket sweep
            nd = len(cfg.radix_bits_sweep)
            Bk = cfg.radix_bkt_bins
            rd_bkt_hist_p = np.zeros((nd, L, Bk), dtype=np.int64)
            rd_bkt_hist_d = np.zeros((nd, L, Bk), dtype=np.int64)
            rd_bkt_zero_p = np.zeros((nd, L), dtype=np.int64)
            rd_bkt_zero_d = np.zeros((nd, L), dtype=np.int64)
            rd_bkt_count_p = np.zeros((nd, L), dtype=np.int64)
            rd_bkt_count_d = np.zeros((nd, L), dtype=np.int64)
            rd_bkt_sum_p = np.zeros((nd, L), dtype=np.int64)
            rd_bkt_sum_d = np.zeros((nd, L), dtype=np.int64)
            rd_bkt_sum_sq_p = np.zeros((nd, L), dtype=np.int64)
            rd_bkt_sum_sq_d = np.zeros((nd, L), dtype=np.int64)
            rd_bkt_max_p = np.zeros((nd, L), dtype=np.int64)
            rd_bkt_max_d = np.zeros((nd, L), dtype=np.int64)
            # Reservoirs
            gap_sample_p = gap_sample_d = None
            sample_p = sample_d = None
            tk_sample_p = tk_sample_d = None
            if cfg.sample_k > 0:
                K = cfg.sample_k
                gap_sample_p = np.full((L, K), np.nan, dtype=np.float32)
                gap_sample_d = np.full((L, K), np.nan, dtype=np.float32)
                sample_p = np.full((L, K), np.nan, dtype=np.float32)
                sample_d = np.full((L, K), np.nan, dtype=np.float32)
                tk_sample_p = np.full((L, K), np.nan, dtype=np.float32)
                tk_sample_d = np.full((L, K), np.nan, dtype=np.float32)

            for (layer, phase), (sA, sB, sC, sD) in self.state.items():
                i = idx[layer]
                if phase == "prefill":
                    A["hist_p"][i] = sA.hist
                    gap_above_p[i] = sA.above; gap_count_p[i] = sA.count
                    gap_skipped_p[i] = sA.skipped
                    gap_sum_p[i] = sA.sum_; gap_sum_sq_p[i] = sA.sum_sq
                    gap_min_p[i] = sA.min_; gap_max_p[i] = sA.max_
                    B["hist_p"][i] = sB.hist
                    count_p[i] = sB.count
                    below_p[i] = sB.below; above_p[i] = sB.above
                    sum_p[i] = sB.sum_; sum_sq_p[i] = sB.sum_sq
                    min_p[i] = sB.min_; max_p[i] = sB.max_
                    tk_count_p[i] = sC.count
                    tk_sum_p[i] = sC.sum_; tk_sum_sq_p[i] = sC.sum_sq
                    tk_min_p[i] = sC.min_; tk_max_p[i] = sC.max_
                    if cfg.sample_k > 0:
                        if sA.reservoir is not None: gap_sample_p[i] = sA.reservoir.array()
                        if sB.reservoir is not None: sample_p[i] = sB.reservoir.array()
                        if sC.reservoir is not None: tk_sample_p[i] = sC.reservoir.array()
                    for di in range(nd):
                        rd_bkt_hist_p[di, i]   = sD.bkt_hist[di]
                        rd_bkt_zero_p[di, i]   = sD.bkt_zero[di]
                        rd_bkt_count_p[di, i]  = sD.bkt_count[di]
                        rd_bkt_sum_p[di, i]    = sD.bkt_sum[di]
                        rd_bkt_sum_sq_p[di, i] = sD.bkt_sum_sq[di]
                        rd_bkt_max_p[di, i]    = sD.bkt_max[di]
                elif phase == "decode":
                    A["hist_d"][i] = sA.hist
                    gap_above_d[i] = sA.above; gap_count_d[i] = sA.count
                    gap_skipped_d[i] = sA.skipped
                    gap_sum_d[i] = sA.sum_; gap_sum_sq_d[i] = sA.sum_sq
                    gap_min_d[i] = sA.min_; gap_max_d[i] = sA.max_
                    B["hist_d"][i] = sB.hist
                    count_d[i] = sB.count
                    below_d[i] = sB.below; above_d[i] = sB.above
                    sum_d[i] = sB.sum_; sum_sq_d[i] = sB.sum_sq
                    min_d[i] = sB.min_; max_d[i] = sB.max_
                    tk_count_d[i] = sC.count
                    tk_sum_d[i] = sC.sum_; tk_sum_sq_d[i] = sC.sum_sq
                    tk_min_d[i] = sC.min_; tk_max_d[i] = sC.max_
                    if cfg.sample_k > 0:
                        if sA.reservoir is not None: gap_sample_d[i] = sA.reservoir.array()
                        if sB.reservoir is not None: sample_d[i] = sB.reservoir.array()
                        if sC.reservoir is not None: tk_sample_d[i] = sC.reservoir.array()
                    for di in range(nd):
                        rd_bkt_hist_d[di, i]   = sD.bkt_hist[di]
                        rd_bkt_zero_d[di, i]   = sD.bkt_zero[di]
                        rd_bkt_count_d[di, i]  = sD.bkt_count[di]
                        rd_bkt_sum_d[di, i]    = sD.bkt_sum[di]
                        rd_bkt_sum_sq_d[di, i] = sD.bkt_sum_sq[di]
                        rd_bkt_max_d[di, i]    = sD.bkt_max[di]

            payload = dict(
                layer_ids=np.array(layer_ids, dtype=np.int32),
                k_index_topk=np.int32(self.seen_k),
                gap_window=np.int32(cfg.gap_window),
                gap_eps_norm=np.float32(cfg.eps_norm),
                # ---- Stream A ----
                gap_bins=np.int32(cfg.gap_bins),
                gap_range_lo=np.float32(0.0),
                gap_range_hi=np.float32(cfg.gap_range_hi),
                gap_hist_prefill=A["hist_p"], gap_hist_decode=A["hist_d"],
                gap_above_prefill=gap_above_p, gap_above_decode=gap_above_d,
                gap_count_prefill=gap_count_p, gap_count_decode=gap_count_d,
                gap_skipped_prefill=gap_skipped_p, gap_skipped_decode=gap_skipped_d,
                gap_sum_prefill=gap_sum_p, gap_sum_decode=gap_sum_d,
                gap_sum_sq_prefill=gap_sum_sq_p, gap_sum_sq_decode=gap_sum_sq_d,
                gap_min_prefill=gap_min_p, gap_min_decode=gap_min_d,
                gap_max_prefill=gap_max_p, gap_max_decode=gap_max_d,
                # ---- Stream B ----
                bins=np.int32(cfg.bins),
                range_lo=np.float32(cfg.range_lo),
                range_hi=np.float32(cfg.range_hi),
                hist_prefill=B["hist_p"], hist_decode=B["hist_d"],
                count_prefill=count_p, count_decode=count_d,
                below_prefill=below_p, below_decode=below_d,
                above_prefill=above_p, above_decode=above_d,
                sum_prefill=sum_p, sum_decode=sum_d,
                sum_sq_prefill=sum_sq_p, sum_sq_decode=sum_sq_d,
                min_prefill=min_p, min_decode=min_d,
                max_prefill=max_p, max_decode=max_d,
                # ---- Stream C ----
                topk_count_prefill=tk_count_p, topk_count_decode=tk_count_d,
                topk_sum_prefill=tk_sum_p, topk_sum_decode=tk_sum_d,
                topk_sum_sq_prefill=tk_sum_sq_p, topk_sum_sq_decode=tk_sum_sq_d,
                topk_min_prefill=tk_min_p, topk_min_decode=tk_min_d,
                topk_max_prefill=tk_max_p, topk_max_decode=tk_max_d,
                # ---- Stream D — radix boundary-bucket sweep ----
                # Arrays are [n_d_values, L, ...]; metadata tells which d values.
                radix_bits_sweep=np.array(cfg.radix_bits_sweep, dtype=np.int32),
                radix_bkt_bins=np.int32(cfg.radix_bkt_bins),
                radix_bkt_hist_prefill=rd_bkt_hist_p,    # [nd, L, Bk]
                radix_bkt_hist_decode=rd_bkt_hist_d,
                radix_bkt_zero_prefill=rd_bkt_zero_p,    # [nd, L]
                radix_bkt_zero_decode=rd_bkt_zero_d,
                radix_bkt_count_prefill=rd_bkt_count_p,
                radix_bkt_count_decode=rd_bkt_count_d,
                radix_bkt_sum_prefill=rd_bkt_sum_p,
                radix_bkt_sum_decode=rd_bkt_sum_d,
                radix_bkt_sum_sq_prefill=rd_bkt_sum_sq_p,
                radix_bkt_sum_sq_decode=rd_bkt_sum_sq_d,
                radix_bkt_max_prefill=rd_bkt_max_p,
                radix_bkt_max_decode=rd_bkt_max_d,
            )
            if cfg.sample_k > 0:
                payload["sample_k"] = np.int32(cfg.sample_k)
                payload["gap_sample_prefill"] = gap_sample_p
                payload["gap_sample_decode"] = gap_sample_d
                payload["sample_prefill"] = sample_p
                payload["sample_decode"] = sample_d
                payload["topk_sample_prefill"] = tk_sample_p
                payload["topk_sample_decode"] = tk_sample_d

            tmp = self.dump_path.with_suffix(".npz.tmp")
            with open(tmp, "wb") as f:
                np.savez(f, **payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.dump_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_recorder: _IndexerRecorder | None = None
_init_lock = threading.Lock()


def record(
    k_cache_prefix: str,
    phase: str,
    logits: torch.Tensor,
    index_topk: int,
    key_valid_starts: torch.Tensor,
    key_valid_ends: torch.Tensor,
) -> None:
    if _CONFIG is None:
        return
    global _recorder
    if _recorder is None:
        with _init_lock:
            if _recorder is None:
                _recorder = _IndexerRecorder(_CONFIG)
    _recorder.record(
        k_cache_prefix, phase, logits, index_topk,
        key_valid_starts, key_valid_ends,
    )
