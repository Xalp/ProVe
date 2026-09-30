# Copyright 2026 The ProVe Authors
# SPDX-License-Identifier: Apache-2.0
"""In-process verifier for ProVe (MarsVerifierStream): a second sglang runner beside the proposer.

The verifier (a smaller stock AR model, Qwen2.5-3B/7B-Instruct in the paper) is served
INSIDE the proposer's process on the same GPU(s), with its own weights, KV pool and CUDA
graphs, so scoring a proposed block costs one graph replay (about 2.6 ms measured) instead
of a transformers forward. It is multi-request and batched: per-request verifier KV (one
request row per proposer request, no owner switching / no refeed) and BATCHED scoring —
all requests' proposed blocks are verified in ONE draft-extend replay per proposer cycle
(bs ladder [1..16]), so verify cost stays ~flat in the number of concurrent requests.

Recipe:
  - plain TpModelWorker(is_draft_worker=True, own pools, dllm_algorithm=None): eager
    extends are vanilla ragged+paged chunked-prefill — prefix-correct by construction;
  - EAGLEDraftExtendCudaGraphRunner in DRAFT_EXTEND_V2 (shim exposes draft_runner, NOT
    model_runner), FA3 backend, fixed 31 tokens/request, all-row fp32 logits;
  - built EAGERLY at tp_worker._init_dllm_algorithm (proposer graphs captured, zero
    replays); shared-ServerArgs fields patched and restored around construction;
  - pool slot 0 is reserved at init: ladder-padded replay rows write their KV there.

Protocol consumed by MarsVerifierStream:
  batched_score(items) with items=[{rpx, feed, need, targets, temp}] -> [p_tilde|None]
  crop(rpx, L), release(rpx); legacy score/rewind/owner kept for bs=1 compatibility.

SIZING THE VERIFIER'S KV POOL (read this before touching mem_fraction).
sglang does NOT interpret mem_fraction_static as "give this runner that much memory". In
model_runner_kv_cache_mixin.profile_max_num_token it reads:

    rest_memory        = available_gpu_memory - total_gpu_memory * (1 - mem_fraction_static)
    max_total_num_tokens = rest_memory_bytes // cell_size

and the trap is the NAME "total_gpu_memory": model_runner.py:603 passes
min_per_gpu_memory, i.e. the FREE memory measured when THIS runner started, not the
card's capacity. With F = free-at-init (what the proposer left over after its weights,
KV pool and graphs) and W = what the verifier's weights cost per rank, the whole formula
collapses to

    pool_bytes = F * mem_fraction_static - W

so the fraction is a share of WHAT WAS LEFT OVER for this runner, which is a different
amount of memory beside every different proposer. Measured: F~22.5 GB next to a small
(4B-class) proposer gives 0.10 -> 8029 tokens and 0.20 -> 27157 tokens, while F~16.8 GB
next to the 32B proposer gives 0.10 -> 760 tokens — below a single 300-token prompt.
batched_score then hit _PoolExhausted and silently dropped every request to y1-only
(accept 3.68 -> 1.51, theta inert). That cost a full day of debugging because nothing in
the log said "starved".

So the knob here is ABSOLUTE: inproc_pool_gb ("the verifier's KV pool should be about
this many GB"; default 2.0). The fraction is derived by inverting the formula above
against live torch.cuda.mem_get_info numbers, which makes one setting portable across
proposer sizes and GPU sizes. An explicit inproc_mem_fraction still pins the raw sglang
fraction (the paper's campaign pinned 0.18 for the 3B verifier with the 32B proposer at
--mem-fraction-static 0.70, and 0.32 / 0.55 for the 7B verifier; TP=2 on 80 GB cards).
A pinned fraction whose pool comes out <= 0 is refused at start-up with the arithmetic.
On top of that:
  - a boot CAPACITY CHECK compares the realized pool against what batched_score can
    actually hold (max_cache per request x the server's max_running_requests) and shouts
    at WARNING with the arithmetic and the remedy;
  - a per-round ADMISSION CHECK skips individual requests that do not fit instead of
    letting one allocation failure abort the whole batched round;
  - every degrade-to-y1-only is COUNTED and surfaced (self.stats / cum_stats_str(),
    logged next to the algorithm's own cum-stats line), so a starved run reports
    "N degraded" instead of quietly looking like a weak model.
The server log must contain "[InprocVerifier] injected into dllm algorithm". If the build
fails, maybe_build() stops the server with the reason; `inproc_fallback: true` in the
yaml (or MARSV_INPROC_FALLBACK=1) restores the old behaviour of falling back to the slow
transformers verifier chain with only a WARNING.
"""
import gc
import logging
import os
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

_BLK = int(os.environ.get("MARSV_BLK", "31"))   # tokens per graph round = speculative_num_steps + 1
# NOTE: feed = tokens realized since the last call + the chain being scored, so a _BLK below
# about 24 pushes the long feeds onto the eager priming path and costs more than the rows save.
_LADDER = [1, 2, 4, 8, 16]
_VTOPK = 8                      # candidates handed back for the p_tilde * q rescoring
_GB = float(1 << 30)

# Absolute default size of the verifier's own KV pool. 2 GB is ~58k tokens of
# Qwen2.5-3B-Instruct KV (36 layers x 2 kv heads x 128 dim x 2 (K,V) x 2 bytes = 36 KiB
# per token) or ~37k tokens of Qwen2.5-7B-Instruct (28 x 4 x 128 x 2 x 2 = 56 KiB), i.e.
# more than a dozen full-length (4096) request states, and it costs the proposer 2 GB
# whether the proposer is 7B or 32B.
_DEFAULT_POOL_GB = 2.0
# Kept free ON TOP of the verifier's weights and pool, because everything that is
# allocated AFTER the pool comes out of it: the runner's own attention-backend
# workspace plus the FlashAttentionBackend built here (~0.35 GB measured) and the
# draft-extend graph capture (0.29 GB measured for bs=[1,2,4,8,16]). 0.75 GB was not
# enough — a squeezed card died with "Capture cuda graph failed: CUDA out of memory,
# tried to allocate 288 MiB" — so keep a full GB. On a card with room this costs
# nothing: the pool is sized absolutely and the reserve only binds when memory is tight.
_DEFAULT_RESERVE_GB = 1.0
_FRACTION_FLOOR, _FRACTION_CEIL = 0.02, 0.98


class _ReqState(SimpleNamespace):
    pass


class _PoolExhausted(Exception):
    pass


def _shim_evict(*_a, **_kw):
    """tree_cache.evict() stand-in for the fake cache _prime hands to ScheduleBatch."""
    raise _PoolExhausted()


def _ckpt_weight_gb(path: str, fp8: bool = False) -> float:
    """Bytes the verifier's weights will take on the card, from the files on disk.

    Used to predict how much of today's free memory is gone by the time sglang profiles
    the pool (the profile happens AFTER load_model). Overestimating is safe — it only
    shrinks the pool a little — so prefer the larger of the safetensors/bin shards and
    fall back to a 3B-class guess when the path is unreadable (a Hugging Face id that
    was not downloaded first; serve.sh downloads ids before it writes the yaml).
    """
    best = 0
    try:
        for exts in ((".safetensors",), (".bin", ".pt", ".pth")):
            tot = 0
            for name in os.listdir(path):
                if name.endswith(exts):
                    try:
                        tot += os.path.getsize(os.path.join(path, name))
                    except OSError:
                        pass
            best = max(best, tot)
    except OSError:
        pass
    gb = best / _GB
    if gb <= 0:
        return 6.5                       # unknown ckpt: assume a 3B in bf16 (Qwen2.5-3B = 6.2 GB)
    return gb * 0.55 if fp8 else gb


def _free_total_gb(gpu_id: int = 0) -> tuple:
    """(free, total) GB on this rank's device, matching what sglang will profile.

    get_available_gpu_memory() empties the torch cache first and takes the MIN across
    ranks when the world is distributed; mirror both so a TP>1 verifier sizes its pool
    identically on every rank (mismatched pool sizes across a sharded model are a
    correctness hazard, not just a memory one).
    """
    torch.cuda.empty_cache()
    try:
        free, total = torch.cuda.mem_get_info(gpu_id)
    except Exception:
        free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / _GB, total / _GB
    try:
        from sglang.srt.distributed import get_world_group

        wg = get_world_group()
        if wg is not None and wg.world_size > 1:
            t = torch.tensor([free_gb, total_gb], dtype=torch.float64)
            torch.distributed.all_reduce(
                t, op=torch.distributed.ReduceOp.MIN, group=wg.cpu_group)
            free_gb, total_gb = float(t[0]), float(t[1])
    except Exception:
        pass
    return free_gb, total_gb


class InprocVerifier:

    def __init__(self, target_tp_worker, verifier_path: str,
                 mem_fraction: Optional[float] = None, max_cache: int = 4096,
                 fp8: bool = False, pool_gb: Optional[float] = None,
                 reserve_gb: Optional[float] = None,
                 expect_concurrency: int = 1, allow_rebuild: bool = True):
        t0 = time.perf_counter()
        self.chunk = _BLK
        self.max_cache = int(max_cache)
        self.max_cache_requested = int(max_cache)
        self.valid_len = 0          # legacy single-owner shim
        self.owner = None
        self._st: Dict[int, _ReqState] = {}
        self._algo_stats = None     # set by maybe_build; mirror of the degrade counters
        self.stats = {
            "score_calls": 0, "items": 0, "degraded": 0,
            "deg_too_long": 0, "deg_no_room": 0, "deg_prime_pool": 0,
            "deg_round_pool": 0, "deg_empty_feed": 0,
        }

        sa = target_tp_worker.server_args
        self.gpu_id = target_tp_worker.gpu_id
        self.verifier_path = verifier_path
        self.fp8 = bool(fp8)
        self.pinned_fraction = (float(mem_fraction) if mem_fraction is not None else None)
        self.pool_gb = float(pool_gb if pool_gb is not None else _DEFAULT_POOL_GB)
        self.reserve_gb = float(
            reserve_gb if reserve_gb is not None else _DEFAULT_RESERVE_GB)
        self.expect_concurrency = max(1, int(expect_concurrency or 1))
        self._last_tokens = {}      # rpx -> the verifier's argmax id per scored row
        self._last_topk = {}        # rpx -> (ids, probs) of the verifier's top-K per row
        self._pinned = {}           # (name, dtype) -> pinned host staging buffer
        # the verifier is sharded over the same TP group as the target, so each rank
        # only pays weights/tp; +0.2 GB covers the load-time buffers that never show up
        # in the checkpoint's file sizes
        tp_size = max(1, int(getattr(sa, "tp_size", 1) or 1))
        self.weights_gb = _ckpt_weight_gb(verifier_path, self.fp8) / tp_size + 0.20
        # additive correction to the weights/overhead estimate above; learned from the
        # realized pool of the first build if we rebuild
        self._overhead_corr_gb = 0.0
        self._rebuilt = False

        self._sa = sa
        self._target = target_tp_worker
        self._patch = {
            "speculative_draft_model_path": verifier_path,
            "context_length": None,     # keep the verifier's native window
            "disable_cuda_graph": True,
            "dllm_algorithm": None,     # plain runner = vanilla extend semantics
            "dllm_algorithm_config": None,
            "mem_fraction_static": None,    # filled per build attempt
            "quantization": ("fp8" if fp8 else None),
        }

        frac = self._plan_fraction()
        self.tp = self._make_worker(frac)
        self.mem_fraction = frac
        self.mr = self.tp.model_runner
        self.device = self.mr.device
        logger.info(f"[InprocVerifier] runner up in {time.perf_counter()-t0:.1f}s "
                    f"max_tokens={self.mr.max_total_num_tokens} "
                    f"mem_fraction_static={frac:.4f}")

        force_rebuild = os.environ.get("MARSV_INPROC_FORCE_REBUILD") == "1"
        allow_rebuild = (allow_rebuild
                         and os.environ.get("MARSV_INPROC_NO_REBUILD") != "1")
        self._capacity_check(allow_rebuild=allow_rebuild, force_rebuild=force_rebuild)

        # ladder-padded replay rows dump KV into slot 0 — reserve it forever
        self._slot0 = self.mr.token_to_kv_pool_allocator.alloc(1)

        t1 = time.perf_counter()
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend)
        from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
            EAGLEDraftExtendCudaGraphRunner)
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        gpatch = {
            "speculative_num_steps": _BLK - 1,      # -> num_tokens_per_bs = 31
            "speculative_num_draft_tokens": _BLK,   # gates FA3's draft_extend buffers
            "speculative_eagle_topk": 1,
            "cuda_graph_bs": list(_LADDER),         # BATCHED verify: the real ladder
            "disable_cuda_graph": False,
            "disable_cuda_graph_padding": False,
            "enable_torch_compile": False,
        }
        gsaved = {k: getattr(sa, k) for k in gpatch}
        for k, v in gpatch.items():
            setattr(sa, k, v)
        try:
            self.extend_backend = FlashAttentionBackend(self.mr, skip_prefill=False)
            shim = SimpleNamespace(
                draft_runner=self.mr,               # V2 selector: no .model_runner
                draft_extend_attn_backend=self.extend_backend,
                speculative_algorithm=SpeculativeAlgorithm.from_string("STANDALONE"),
            )
            self.graph_runner = EAGLEDraftExtendCudaGraphRunner(shim)
        finally:
            for k, v in gsaved.items():
                setattr(sa, k, v)
        logger.info(f"[InprocVerifier] draft-extend graphs captured "
                    f"(bs={_LADDER}) in {time.perf_counter()-t1:.1f}s")
        _avail = getattr(self.mr.token_to_kv_pool_allocator, "available_size", lambda: -1)
        logger.info(f"[InprocVerifier] pool avail post-capture={_avail()}")
        self._self_test()
        logger.info(f"[InprocVerifier] pool avail post-selftest={_avail()}")
        # the self-test drives the same degrade paths as production; do not let its
        # book-keeping show up as deployment starvation
        for k in self.stats:
            self.stats[k] = 0
        self._booted = True

    # ---------------- pool sizing ----------------

    def _plan_fraction(self) -> float:
        """Pick mem_fraction_static so the pool lands on the ABSOLUTE target.

        Inverting the real formula (see the module docstring): the runner will size its
        pool as F*frac - W, with F the free memory it measures at its own init — which
        is the free memory RIGHT NOW, a moment before we construct it — and W what its
        weights cost. So

            frac = (pool_gb + W) / F,   capped so that F*(1-frac) >= reserve_gb.

        W is estimated from the checkpoint on disk (per TP rank); any error is measured
        against the realized pool afterwards and can be corrected by one rebuild.
        """
        free_gb, total_gb = _free_total_gb(self.gpu_id)
        self._free_gb_at_plan, self._total_gb = free_gb, total_gb
        w = self.weights_gb + self._overhead_corr_gb
        self._frac_was_capped = True
        if self.pinned_fraction is not None:
            pred = free_gb * self.pinned_fraction - w
            logger.info(
                f"[InprocVerifier] pool sizing: mem_fraction_static PINNED at "
                f"{self.pinned_fraction:.4f} -> predicted pool {pred:.2f} GB "
                f"(free-for-this-runner {free_gb:.2f} GB, weights ~{w:.2f} GB, card "
                f"{total_gb:.2f} GB). inproc_pool_gb ignored. NOTE: this fraction is a "
                f"share of what the PROPOSER left over, so it does not port across "
                f"proposer or GPU sizes — drop it and let inproc_pool_gb size the pool.")
            if pred <= 0.0:
                # Building anyway makes sglang raise its generic "Not enough memory" from
                # deep inside the runner; say what is actually wrong and how to fix it.
                raise RuntimeError(
                    f"inproc_mem_fraction is PINNED at {self.pinned_fraction:.3f} but the "
                    f"verifier's KV pool would be {pred:.2f} GB: the pool is "
                    f"free_after_proposer * fraction - verifier_weights = {free_gb:.2f} GB * "
                    f"{self.pinned_fraction:.3f} - {w:.2f} GB (weights per rank + load "
                    f"buffers). Remove inproc_mem_fraction from the yaml and let "
                    f"inproc_pool_gb size the pool (portable across proposer and GPU "
                    f"sizes), or raise the fraction to at least "
                    f"{(w + 0.5) / max(free_gb, 1e-6):.3f}, or lower the proposer's "
                    f"--mem-fraction-static "
                    f"({getattr(self._sa, 'mem_fraction_static', '?')}) so that more "
                    f"memory is left for the verifier.")
            return float(self.pinned_fraction)
        want = self.pool_gb
        frac_want = (want + w) / max(free_gb, 1e-6)
        frac_cap = 1.0 - (self.reserve_gb / max(free_gb, 1e-6))
        frac = min(frac_want, frac_cap)
        frac_c = min(max(frac, _FRACTION_FLOOR), _FRACTION_CEIL)
        self._frac_was_capped = (frac_c != frac_want)
        pred = free_gb * frac_c - w
        if pred < want - 0.05:
            logger.warning(
                f"[InprocVerifier] POOL TARGET NOT REACHABLE: asked for {want:.2f} GB "
                f"of verifier KV, but this runner only has {free_gb:.2f} GB of free "
                f"memory to work with and needs {w:.2f} GB of it for weights plus "
                f"{self.reserve_gb:.2f} GB of reserve — the pool will be ~{pred:.2f} GB. "
                f"The TARGET's --mem-fraction-static is what is eating the card; lower "
                f"it (or lower inproc_pool_gb / inproc_reserve_gb).")
        if pred <= 0.0:
            # Building anyway means sglang raises its generic "Not enough memory.
            # Please try to increase --mem-fraction-static" from four frames down,
            # which points at the wrong knob entirely. Say what is actually wrong.
            raise RuntimeError(
                f"verifier KV pool would be {pred:.2f} GB: this runner has only "
                f"{free_gb:.2f} GB free and the verifier needs ~{w:.2f} GB of weights plus "
                f"{self.reserve_gb:.2f} GB of workspace/graph reserve before any KV at "
                f"all. The TARGET's --mem-fraction-static "
                f"({getattr(self._sa, 'mem_fraction_static', '?')}) leaves nothing for "
                f"an in-process verifier — lower it by at least "
                f"{(w + self.reserve_gb + 0.5 - free_gb) / max(total_gb, 1):.3f}")
        logger.info(
            f"[InprocVerifier] pool sizing: want {want:.2f} GB KV | free-for-this-runner "
            f"{free_gb:.2f} GB (card {total_gb:.2f} GB), weights/rank ~"
            f"{self.weights_gb:.2f} GB, corr {self._overhead_corr_gb:+.2f} GB, reserve "
            f"{self.reserve_gb:.2f} GB -> mem_fraction_static={frac_c:.4f} "
            f"(predicted pool {pred:.2f} GB)"
            + ("" if frac == frac_c else f" [clamped from {frac:.4f}]"))
        return frac_c

    def _make_worker(self, frac: float):
        from sglang.srt.managers.tp_worker import TpModelWorker

        sa, tw = self._sa, self._target
        patch = dict(self._patch)
        patch["mem_fraction_static"] = float(frac)
        saved = {k: getattr(sa, k) for k in patch}
        for k, v in patch.items():
            setattr(sa, k, v)
        try:
            return TpModelWorker(
                server_args=sa,
                gpu_id=tw.gpu_id,
                tp_rank=tw.tp_rank,
                pp_rank=0,
                dp_rank=tw.dp_rank,
                moe_ep_rank=tw.moe_ep_rank,
                attn_cp_rank=getattr(tw, "attn_cp_rank", 0),
                moe_dp_rank=getattr(tw, "moe_dp_rank", 0),
                nccl_port=tw.nccl_port,
                is_draft_worker=True,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
            )
        finally:
            for k, v in saved.items():
                setattr(sa, k, v)

    def _pool_bytes(self) -> int:
        """Bytes actually held by the verifier's KV pool (0 if it cannot be read)."""
        try:
            pool = self.mr.token_to_kv_pool
            n = pool.get_kv_size_bytes()
            return int(sum(n)) if isinstance(n, (tuple, list)) else int(n)
        except Exception:
            return 0

    def _cell_size(self) -> int:
        tok = max(1, int(getattr(self.mr, "max_total_num_tokens", 0)))
        b = self._pool_bytes()
        return int(b // tok) if b else 0

    def _capacity_check(self, allow_rebuild: bool, force_rebuild: bool = False):
        """Does the pool hold what batched_score will actually ask of it?

        THE point of this method: starvation next to the 32B proposer was invisible. A pool of 760 tokens
        next to prompts of 300 produced no error, just a collapsed accept length. Every
        number that would have identified it in one glance is logged here, and anything
        short is a WARNING with the remedy attached.
        """
        pool_tok = int(getattr(self.mr, "max_total_num_tokens", 0))
        cell = self._cell_size()
        pool_gb = (pool_tok * cell) / _GB if cell else 0.0
        per_req = self.max_cache + _BLK      # ids + one padded graph round
        conc = self.expect_concurrency
        fits = (pool_tok - 1) // per_req if per_req > 0 else 0
        need_all = conc * per_req + 1
        need_gb = (need_all * cell) / _GB if cell else 0.0
        head = (f"[InprocVerifier] KV CAPACITY: pool={pool_tok} tok "
                f"({pool_gb:.2f} GB, cell={cell} B/tok) | max_cache={self.max_cache}"
                f"(+{_BLK} pad) -> {fits} full-length request(s) fit; "
                f"max_running_requests={conc} would need {need_all} tok "
                f"({need_gb:.2f} GB)")

        if fits >= 1 and fits >= conc:
            logger.info(head + " — OK")
            return

        if fits < 1:
            logger.warning(
                head + " — STARVED: the pool cannot hold even ONE request at "
                f"max_cache={self.max_cache}. batched_score will hit _PoolExhausted and "
                f"silently degrade requests to y1-only (accept length collapses to ~1.5 "
                f"and theta stops mattering). Remedy: set inproc_pool_gb >= "
                f"{max(need_gb / max(conc, 1) * 1.1, 0.25):.2f} in the algorithm yaml, "
                f"or lower the TARGET's --mem-fraction-static so there is memory left, "
                f"or lower inproc_max_cache.")
            if (allow_rebuild and self.pinned_fraction is None
                    and not self._rebuilt and self._rebuild(pool_tok, cell, per_req)):
                return self._capacity_check(allow_rebuild=False)
            # cannot (or must not) rebuild: make the truth the configuration, so the
            # behaviour is a predictable per-request skip instead of pool thrashing
            new_cache = max(0, pool_tok - 1 - _BLK)
            if new_cache < self.max_cache:
                logger.warning(
                    f"[InprocVerifier] lowering max_cache {self.max_cache} -> "
                    f"{new_cache} to match the pool that actually exists. Requests "
                    f"longer than that are now SKIPPED (counted as deg_too_long) "
                    f"instead of thrashing the allocator."
                    + ("" if new_cache >= 512 else
                       " At this size essentially every request will degrade to "
                       "y1-only — the verifier is effectively off."))
                self.max_cache = new_cache
        else:
            logger.warning(
                head + f" — UNDERSIZED for full concurrency: only {fits} of {conc} "
                f"concurrent requests can hold a full {self.max_cache}-token state. "
                f"Beyond that, requests are skipped (deg_no_room) and degrade to "
                f"y1-only. Remedy: inproc_pool_gb >= {need_gb * 1.05:.2f} for full "
                f"coverage, or accept it if real sequences stay well under "
                f"{self.max_cache} tokens.")

        if force_rebuild and self.pinned_fraction is None and not self._rebuilt:
            logger.warning("[InprocVerifier] MARSV_INPROC_FORCE_REBUILD=1: exercising "
                           "the rebuild path on purpose")
            if self._rebuild(pool_tok, cell, per_req, force=True):
                self._capacity_check(allow_rebuild=False)

    def _rebuild(self, pool_tok: int, cell: int, per_req: int,
                 force: bool = False) -> bool:
        """Tear the runner down and build it again at a corrected fraction.

        Safe to do HERE and only here: no graphs have been captured yet (the
        draft-extend capture happens after this returns) and slot 0 is not yet
        reserved, so the second runner starts from the same clean state the first did.
        The verifier rebuilds in ~1 s. If the rebuild raises, the exception propagates to
        maybe_build, which stops the server with the reason (or, with `inproc_fallback:
        true`, disables the in-process verifier loudly) rather than leaving a half-built
        one behind.
        """
        realized_gb = (pool_tok * cell) / _GB if cell else 0.0
        # what the first attempt got wrong: our estimate of the memory that disappears
        # between the measurement and sglang's profile. Carrying it as an additive
        # correction makes the second attempt land on the target. Only meaningful when
        # the first fraction was NOT clamped (a clamp is a memory limit, not an error).
        corr = self.pool_gb - realized_gb
        if getattr(self, "_frac_was_capped", False):
            corr = 0.0
        min_need = (per_req + 1) * cell / _GB * 1.10 if cell else 0.25
        want = self.pool_gb if force else max(self.pool_gb, min_need)
        free_gb, _total_gb = _free_total_gb(self.gpu_id)
        # everything this runner holds (pool AND weights) comes back on teardown, and
        # the retry pays the same weights again, so the weights cancel: the biggest pool
        # a retry can reach is (free now + the pool we are about to drop) - reserve.
        reachable = free_gb + realized_gb - self.reserve_gb
        if want > reachable and not force:
            if min_need > reachable:
                logger.warning(
                    f"[InprocVerifier] NOT rebuilding: even a perfect fraction only "
                    f"reaches {reachable:.2f} GB of KV here and one request at "
                    f"max_cache={self.max_cache} needs {min_need:.2f} GB. The card is "
                    f"genuinely full — lower the TARGET's --mem-fraction-static.")
                return False
            # the full target does not fit but a useful pool does: take what is there
            want = reachable * 0.95
            logger.warning(f"[InprocVerifier] rebuild target trimmed to {want:.2f} GB "
                           f"(all that is reachable; {self.pool_gb:.2f} GB was asked)")
        logger.warning(
            f"[InprocVerifier] rebuilding the verifier runner: realized "
            f"{realized_gb:.2f} GB vs {self.pool_gb:.2f} GB asked "
            f"(overhead estimate off by {corr:+.2f} GB); retry target {want:.2f} GB")
        before = _free_total_gb(self.gpu_id)[0]
        try:
            self.mr = None
            self.device = None
            tp, self.tp = self.tp, None
            del tp
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"[InprocVerifier] teardown raised {e!r}; continuing")
        after = _free_total_gb(self.gpu_id)[0]
        logger.warning(f"[InprocVerifier] teardown freed {after-before:.2f} GB "
                       f"(free {before:.2f} -> {after:.2f} GB)")
        self._rebuilt = True
        self._overhead_corr_gb += corr
        self.pool_gb = want
        t0 = time.perf_counter()
        frac = self._plan_fraction()        # re-measures free memory post-teardown
        self.tp = self._make_worker(frac)
        self.mem_fraction = frac
        self.mr = self.tp.model_runner
        self.device = self.mr.device
        logger.warning(f"[InprocVerifier] rebuilt in {time.perf_counter()-t0:.1f}s "
                       f"max_tokens={self.mr.max_total_num_tokens} "
                       f"mem_fraction_static={frac:.4f}")
        return True

    # ---------------- degrade accounting ----------------

    def cum_stats_str(self) -> str:
        s = self.stats
        d, n = s["degraded"], max(1, s["items"])
        # show the configured max_cache too when the boot check had to lower it —
        # otherwise a wall of deg_too_long looks unexplained
        mc = (f"{self.max_cache}" if self.max_cache == self.max_cache_requested
              else f"{self.max_cache}(lowered from {self.max_cache_requested})")
        return (f"degraded={d}/{s['items']} ({d/n:.1%}) "
                f"[too_long={s['deg_too_long']} no_room={s['deg_no_room']} "
                f"prime_pool={s['deg_prime_pool']} round_pool={s['deg_round_pool']} "
                f"empty_feed={s['deg_empty_feed']}] "
                f"pool={getattr(self.mr, 'max_total_num_tokens', -1)}tok "
                f"max_cache={mc} live_states={len(self._st)}")

    def _degrade(self, kind: str, n: int = 1):
        """Every path that drops a request to y1-only goes through here.

        Silence is what made the KV starvation expensive, so this counts, mirrors the
        count into the algorithm's stats dict, and shouts on a 1/10/100/1000 ladder
        (never per-token, which would sit in the decode critical path)."""
        if n <= 0:
            return
        self.stats[kind] = self.stats.get(kind, 0) + n
        self.stats["degraded"] += n
        if self._algo_stats is not None:
            try:
                self._algo_stats["inproc_degraded"] = self.stats["degraded"]
                self._algo_stats["inproc_items"] = self.stats["items"]
            except Exception:
                pass
        d = self.stats["degraded"]
        if d in (1, 10, 100) or d % 1000 == 0:
            logger.warning(f"[InprocVerifier] DEGRADED to y1-only ({kind}): "
                           f"{self.cum_stats_str()}")

    # ---------------- per-request state ----------------

    def _get_st(self, rpx: int) -> _ReqState:
        st = self._st.get(rpx)
        if st is None:
            st = _ReqState(req=None, ids=[], high=0)
            self._st[rpx] = st
        return st

    def _row(self, st: _ReqState):
        return self.mr.req_to_token_pool.req_to_token[st.req.req_pool_idx]

    def vlen(self, rpx: int) -> int:
        st = self._st.get(rpx)
        return len(st.ids) if st else 0

    def crop(self, rpx: int, L: int):
        st = self._st.get(rpx)
        if st is None:
            return
        if st.req is not None and st.high > L:
            row = self._row(st)
            import os as _os
            if _os.environ.get("MARSV_LEDGER"):
                a0 = self.mr.token_to_kv_pool_allocator.available_size()
                self.mr.token_to_kv_pool_allocator.free(row[L:st.high].clone())
                a1 = self.mr.token_to_kv_pool_allocator.available_size()
                logger.info(f"[ledger] crop rpx={rpx} L={L} high={st.high} "
                            f"want_free={st.high-L} got_back={a1-a0}")
            else:
                self.mr.token_to_kv_pool_allocator.free(row[L:st.high].clone())
            st.high = L
        st.ids = st.ids[:L]

    def release(self, rpx: int):
        st = self._st.pop(rpx, None)
        if st is None or st.req is None:
            self._log_cum_stats()
            return
        import os as _os
        if _os.environ.get("MARSV_CYCLE_LOG"):
            logger.info(f"[InprocVerifier] release rpx={rpx} len={len(st.ids)} "
                        f"high={st.high} states_left={len(self._st)}")
        self.crop_state(st, 0)
        self._log_cum_stats()

    def _log_cum_stats(self):
        """Piggyback on the algorithm's per-request cum-stats logging.

        MarsVerifierStream.cleanup_request() calls release() immediately before it logs
        its own "cum stats:" line, so this lands right next to it: a starved run now
        reads "degraded=N/M" in the same place the accept length is read."""
        if not getattr(self, "_booted", False):
            return                       # boot self-test releases are not deployment
        if self.stats["degraded"]:
            logger.warning(f"[InprocVerifier] cum stats: {self.cum_stats_str()}")
        elif os.environ.get("MARSV_LOG"):
            logger.info(f"[InprocVerifier] cum stats: {self.cum_stats_str()}")

    def crop_state(self, st: _ReqState, L: int):
        if st.req is not None and st.high > L:
            row = self._row(st)
            self.mr.token_to_kv_pool_allocator.free(row[L:st.high].clone())
            st.high = L
        st.ids = st.ids[:L]
        if L == 0 and st.req is not None:
            self.mr.req_to_token_pool.free(st.req)
            st.req = None

    # ---------------- eager prime (vanilla extend, arbitrary length) ----------------

    @torch.no_grad()
    def _prime(self, st: _ReqState, part: List[int]):
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        mr = self.mr
        prev = len(st.ids)
        if st.req is None:
            st.req = Req(rid=0, origin_input_text="", origin_input_ids=[],
                         sampling_params=SamplingParams(
                             temperature=0, max_new_tokens=1))
            st.req.logprob_start_len = -1
        req = st.req
        req.fill_ids = st.ids + part
        req.origin_input_ids = req.fill_ids
        req.prefix_indices = (self._row(st)[:prev] if prev else torch.empty(
            (0,), dtype=torch.int64, device=mr.device))
        req.set_extend_input_len(len(part))
        cache = SimpleNamespace(
            page_size=mr.server_args.page_size, device=mr.device,
            token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
            supports_swa=lambda: False, supports_mamba=lambda: False,
            # This shim is not a real radix cache, so there is nothing to evict. When
            # the pool is short, alloc_for_extend calls evict() and previously died on
            # AttributeError, taking the whole scheduler with it. Raise the soft error
            # the batched path already knows how to degrade from instead.
            evict=_shim_evict,
            is_chunk_cache=lambda: False, is_tree_cache=lambda: True)
        batch = ScheduleBatch.init_new(
            reqs=[req], req_to_token_pool=mr.req_to_token_pool,
            token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
            tree_cache=cache, model_config=mr.model_config,
            enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE)
        batch.prepare_for_extend()
        fb = ForwardBatch.init_new(batch.get_model_worker_batch(), mr)
        out = mr.forward(fb)
        torch.cuda.synchronize()
        st.ids.extend(part)
        st.high = len(st.ids)
        # returned only for the numeric self-test, which needs a graph-free reference;
        # every production caller ignores it
        return out

    # ---------------- batched graph round ----------------

    @torch.no_grad()
    def _pin(self, name, n, dtype):
        """Grow-only pinned host buffer, one per (name, dtype). Pinned memory makes the H2D
        copies asynchronous, so the host keeps queueing instead of waiting for each one."""
        key = (name, dtype)
        buf = self._pinned.get(key)
        if buf is None or buf.numel() < n:
            buf = torch.empty(max(n, 64), dtype=dtype, pin_memory=True)
            self._pinned[key] = buf
        return buf

    def _batched_round(self, parts):
        """parts: list of (st, tokens<=31). One DRAFT_EXTEND_V2 replay for all.
        Returns list of fp32 logits rows [len(tokens_i), V]."""
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch, ForwardMode)
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        mr = self.mr
        dev = mr.device
        bs = len(parts)
        all_tokens, all_pos, all_slots = [], [], []
        seq_lens = []
        for st, part in parts:
            pl = len(part)
            prev = len(st.ids)
            if st.high > prev:                  # release last round's pad slots
                row = self._row(st)
                import os as _os
                if _os.environ.get("MARSV_LEDGER"):
                    a0 = mr.token_to_kv_pool_allocator.available_size()
                    mr.token_to_kv_pool_allocator.free(row[prev:st.high].clone())
                    a1 = mr.token_to_kv_pool_allocator.available_size()
                    logger.info(f"[ledger] padrel prev={prev} high={st.high} "
                                f"want={st.high-prev} got={a1-a0}")
                else:
                    mr.token_to_kv_pool_allocator.free(row[prev:st.high].clone())
                st.high = prev
            slots = mr.token_to_kv_pool_allocator.alloc(_BLK)
            if slots is None:
                # pool exhausted: fail SOFT — abort this whole batched round; the
                # caller returns p_tilde=None and every request degrades to y1-only.
                avail = getattr(mr.token_to_kv_pool_allocator, "available_size", lambda: -1)()
                logger.warning(
                    f"[InprocVerifier] KV pool exhausted "
                    f"(states={len(self._st)}, sizes={[len(x.ids) for x in self._st.values()][:12]}, "
                    f"allocator_avail={avail}, pool={getattr(mr, 'max_total_num_tokens', -1)}) "
                    f"— raise inproc_pool_gb or lower inproc_max_cache")
                raise _PoolExhausted()
            row = self._row(st)
            row[prev:prev + _BLK] = slots.to(row.dtype)
            all_tokens.extend(part + [0] * (_BLK - pl))
            all_pos.append(torch.arange(prev, prev + _BLK, dtype=torch.int64,
                                        device=dev))
            all_slots.append(slots)
            seq_lens.append(prev + _BLK)

        # PINNED staging for the three host->device copies of this round. torch.tensor(list,
        # device=cuda) allocates pageable memory, and a pageable H2D blocks the calling thread
        # until the copy retires, so each one used to stall the critical path.
        self._round = (getattr(self, "_round", 0) + 1) & 1        # double buffer
        r = str(self._round)
        ids_h = self._pin("ids" + r, len(all_tokens), torch.int64); ids_h[:len(all_tokens)] = torch.tensor(all_tokens, dtype=torch.int64)
        rpi_h = self._pin("rpi" + r, bs, torch.int32); rpi_h[:bs] = torch.tensor([st.req.req_pool_idx for st, _ in parts], dtype=torch.int32)
        sl_h = self._pin("sl" + r, bs, torch.int32); sl_h[:bs] = torch.tensor(seq_lens, dtype=torch.int32)
        fb = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=bs,
            input_ids=ids_h[:len(all_tokens)].to(dev, non_blocking=True),
            req_pool_indices=rpi_h[:bs].to(dev, non_blocking=True),
            seq_lens=sl_h[:bs].to(dev, non_blocking=True),
            seq_lens_cpu=sl_h[:bs].clone(),
            seq_lens_sum=int(sum(seq_lens)),
            out_cache_loc=torch.cat(all_slots),
            positions=torch.cat(all_pos),
            extend_seq_lens=torch.full((bs,), _BLK, dtype=torch.int32, device=dev),
            extend_seq_lens_cpu=[_BLK] * bs,
            spec_info=EagleDraftInput(
                hidden_states=torch.zeros((bs * _BLK, 1), dtype=mr.dtype,
                                          device=dev),
                accept_length=torch.full((bs,), _BLK, dtype=torch.int32,
                                         device=dev),
                num_tokens_per_req=_BLK,
                num_tokens_for_logprob_per_req=_BLK,
            ),
        )
        assert self.graph_runner.can_run(fb), \
            f"draft-extend graph cannot run bs={bs}"
        out = self.graph_runner.replay(fb)
        logits = out.next_token_logits[:bs * _BLK]
        rows_out = []
        for i, (st, part) in enumerate(parts):
            pl = len(part)
            rows_out.append(logits[i * _BLK:i * _BLK + pl].float().clone())
            st.ids.extend(part)
            # high-water = ALL 31 allocated slots incl. pads — pads above len(ids)
            # must stay in the ledger or they leak 31-pl slots per cycle
            st.high = len(st.ids) + (_BLK - pl)
        return rows_out

    # ---------------- batched scoring API ----------------

    @torch.no_grad()
    def batched_score(self, items):
        """items: [{rpx, feed, need, targets, temp}] -> [p_tilde tensor | None].
        Long feeds (first cycle: whole prompt) are eager-primed per request first;
        the <=31-token tails of ALL items go through ONE graph replay."""
        results = [None] * len(items)
        parts = []
        meta = []
        self.stats["score_calls"] += 1
        self.stats["items"] += len(items)
        # ADMISSION: how many slots can this round actually spend? One request that does
        # not fit used to raise _PoolExhausted from inside the round and take every other
        # request in the batch down with it; checking first turns that into a targeted,
        # counted skip of the request that does not fit.
        try:
            budget = self.mr.token_to_kv_pool_allocator.available_size()
        except Exception:
            budget = None
        for j, it in enumerate(items):
            st = self._get_st(it["rpx"])
            feed = list(it["feed"])
            if len(st.ids) + len(feed) + _BLK > self.max_cache:
                self._degrade("deg_too_long")
                continue                      # too long: caller falls back to y1-only
            if budget is not None:
                # slots this request will take this round, minus the pad slots the round
                # is about to hand back for it
                need = len(feed) + _BLK - max(0, st.high - len(st.ids))
                if need > budget:
                    self._degrade("deg_no_room")
                    continue
                budget -= need
            try:
                if st.req is None and len(feed) <= _BLK:
                    self._prime(st, feed[:1])   # row must exist before a graph round
                    feed = feed[1:]
                elif len(feed) > _BLK:
                    self._prime(st, feed[:-_BLK])
                    feed = feed[-_BLK:]
            except _PoolExhausted:
                # this request cannot be primed; skip it (caller degrades it to
                # y1-only) rather than taking the scheduler down with us
                logger.warning("[InprocVerifier] prime hit an exhausted KV pool; "
                               "request degraded to y1-only")
                self._degrade("deg_prime_pool")
                continue
            if not feed:
                self._degrade("deg_empty_feed")
                continue
            parts.append((st, feed))
            meta.append((j, it, len(feed)))
        if not parts:
            return results
        # cap at the ladder max: split oversized batches into ladder-sized chunks
        rows_all = []
        i = 0
        try:
            while i < len(parts):
                take = min(len(parts) - i, _LADDER[-1])
                rows_all.extend(self._batched_round(parts[i:i + take]))
                i += take
        except _PoolExhausted:
            lost = len(parts) - len(rows_all)
            self._degrade("deg_round_pool", lost)
            rows_all.extend([None] * lost)
        for (j, it, pl), rows in zip(meta, rows_all):
            if rows is None:
                continue
            need = it["need"]
            lo = max(0, pl - need)
            lg = rows[lo:]
            if it.get("temp", 1.0) != 1.0:
                lg = lg / it["temp"]
            lp = torch.log_softmax(lg, dim=-1)
            tgt = torch.tensor(list(it["targets"])[-lg.shape[0]:],
                               dtype=torch.long, device=lg.device)
            results[j] = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).exp()
            # the verifier's OWN argmax at each scored row, conditioned on the same
            # prefix as the score beside it. MarsVerifierStream commits it as one extra
            # token at the rejection position when the yaml sets vtok: true; the tensor
            # is free (one argmax over rows already in registers) and unread otherwise.
            self._last_tokens[it["rpx"]] = lg.argmax(dim=-1)
            # ... and its top-K, so the caller can pick the token that maximizes the GATE
            # score p_tilde * q over a candidate set instead of trusting q alone. K rows of
            # a topk over the vocabulary is the same order of work as the log_softmax above.
            tv, ti = lp.topk(_VTOPK, dim=-1)
            self._last_topk[it["rpx"]] = (ti, tv.exp())
        return results


    # ---------------- ANCHOR ROUND: score + draft at EVERY anchor in ONE replay -----------
    def _abump(self, key):
        """Count into the algorithm's stats dict so the reason survives in the durable log."""
        a = getattr(self, "_algo_stats", None)
        if a is not None:
            a[key] = a.get(key, 0) + 1

    def _alias_child(self, parent_st, L):
        """A throwaway request whose first L token slots POINT AT the parent's KV pages.
        No forward is run: a req_pool row is allocated and the parent's index array is copied,
        so the child sees the realized prefix for free. Only the child's own slots are freed."""
        pool = self.mr.req_to_token_pool
        # ReqToTokenPool.alloc() takes a list[Req] and calls is_dllm()/is_chunked on each, and
        # free() takes a Req and clears req_pool_idx. These throwaway children are not Reqs, so
        # take and return a row index directly -- the same free_slots list the pool itself uses.
        if not pool.free_slots:
            raise _PoolExhausted()
        i = int(pool.free_slots.pop(0))
        pool.req_to_token[i, :L] = self._row(parent_st)[:L]
        return _ReqState(req=SimpleNamespace(req_pool_idx=i), ids=list(parent_st.ids[:L]),
                         high=L, alias_len=L, pool_idx=i)

    def _free_child(self, st):
        """Return the child's OWN KV slots and its row index. The aliased prefix belongs to the
        parent and must never be freed. Uses the high-water mark the round actually reached, so
        an aborted round still returns whatever it managed to allocate."""
        pool = self.mr.req_to_token_pool
        row = pool.req_to_token[st.req.req_pool_idx]
        hi = max(int(getattr(st, "high", st.alias_len)), st.alias_len)
        if hi > st.alias_len:
            self.mr.token_to_kv_pool_allocator.free(row[st.alias_len:hi].clone())
        if st.pool_idx not in pool.free_slots:
            pool.free_slots.append(int(st.pool_idx))

    @torch.no_grad()
    def anchor_round(self, rpx, feed, drafts, mask_id, n_cont=16):
        """ONE graph replay that does the whole anchor search.

        Group 0 scores the proposed chain; group j (j = 1..len(drafts)-1) feeds
        [feed + drafts[:j] + n_cont placeholders] in its OWN row, aliasing the realized
        prefix, so its right-shifted readout is a block draft anchored exactly at j.
        Rows are cheap on a small model, which is the whole point: 16 anchors x 31 rows
        costs one replay instead of a second call plus a crop.

        Returns (p_tilde over drafts[1:], {j: [continuation ids]}) or (None, {}).
        """
        st = self._get_st(rpx)
        feed = list(feed); drafts = list(drafts)
        n = len(drafts)
        if n < 2:
            self._abump("ab_short"); return None, {}
        # PRIME: a state only owns a req_pool row once something has primed it, and the
        # children alias THAT row. batched_score does this inline; forgetting it here made
        # every cycle bail at the first guard (measured: refill_none == cycles, exactly).
        # The round must hold feed + drafts[:-1] for the scoring group and feed + drafts[:j] +
        # placeholders for each child, all within _BLK. Priming down to _BLK-1 (copied from
        # batched_score, where the whole round IS the feed) leaves room == 1 and bails forever.
        keep = max(1, _BLK - n - 1)
        try:
            if len(feed) > keep:
                self._prime(st, feed[:-keep]); feed = feed[-keep:]
            if st.req is None:
                if not feed:
                    return None, {}
                self._prime(st, feed[:1]); feed = feed[1:]
        except _PoolExhausted:
            self._degrade("deg_anchor_prime"); self._abump("ab_prime"); return None, {}
        base_len = len(st.ids)                       # realized prefix already in the parent's KV
        if len(feed) + n - 1 > _BLK:                 # scoring group must fit the captured round
            self._abump("ab_room"); return None, {}
        room = _BLK - len(feed)
        if room < 3:
            self._abump("ab_room"); return None, {}
        anchors = [j for j in range(1, n) if len(feed) + j + 1 <= _BLK]
        anchors = anchors[: _LADDER[-1] - 1]
        # Keep EVERY anchor that fits. An earlier version trimmed to the largest ladder batch
        # size, which capped the list at 7 and pulled the chosen anchor 0.67 slots early on
        # average (measured) -- the refill then replaced big-model drafts that would have
        # verified. disable_cuda_graph_padding is False for this runner, so an off-ladder batch
        # is padded up to the next captured size instead of failing.
        if not anchors:
            self._abump("ab_noanchor"); return None, {}
        parts = [(st, feed + drafts[:-1])]           # group 0: scoring
        kids = []
        try:
            for j in anchors:
                ch = self._alias_child(st, base_len)
                m = min(n_cont, _BLK - len(feed) - j)
                kids.append((ch, j, m))
                parts.append((ch, feed + drafts[:j] + [int(mask_id)] * max(0, m - 1)))
            rows_all = self._batched_round(parts)
        except BaseException as _e:
            for ch, _, _ in kids:
                try: self._free_child(ch)
                except Exception: pass
            if isinstance(_e, _PoolExhausted):
                self._degrade("deg_anchor_pool"); self._abump("ab_pool"); return None, {}
            self._abump("ab_exc_" + type(_e).__name__)
            logger.warning(f"[InprocVerifier] anchor_round aborted: {_e!r}")
            return None, {}
        out = {}
        lp0 = rows_all[0]
        pt = None
        if lp0 is not None:
            lg = lp0[-(n - 1):]
            tgt = torch.tensor(drafts[1:], dtype=torch.long, device=lg.device)
            pt = torch.log_softmax(lg, dim=-1).gather(1, tgt.unsqueeze(1)).squeeze(1).exp()
        for (ch, j, m), rows in zip(kids, rows_all[1:]):
            if rows is not None and m >= 1:
                # right-shifted readout: the last REAL row predicts slot 1, mask rows the rest
                out[j] = rows[-m:].argmax(dim=-1).tolist()
            self._free_child(ch)
        self.stats["score_calls"] += 1
        self._abump("ab_ok")
        return pt, out

    # ---------------- batched DRAFTING API (ReSpec-3: the small model drafts) --------------

    @torch.no_grad()
    def batched_draft(self, items):
        """items: [{rpx, feed, n_draft, mask_id}] -> [list[int] of n_draft greedy draft ids | None].
        The in-process model is a BLOCK-DRAFTING AR model. feed = the tokens committed since its
        last call; n_draft-1 placeholder (mask) tokens are appended. With the right-shifted readout,
        the logits at the last committed token give y_1 and the logits at placeholder i give
        y_{i+1}. The placeholder slots are cropped from the state afterwards, so the cache holds
        committed tokens only. Same admission / priming / degrade rules as batched_score."""
        results = [None] * len(items)
        parts, meta = [], []
        self.stats["score_calls"] += 1
        self.stats["items"] += len(items)
        try:
            budget = self.mr.token_to_kv_pool_allocator.available_size()
        except Exception:
            budget = None
        for j, it in enumerate(items):
            st = self._get_st(it["rpx"])
            n_draft = int(it["n_draft"])
            n_mask = max(0, n_draft - 1)
            feed = list(it["feed"]) + [int(it["mask_id"])] * n_mask
            if len(st.ids) + len(feed) + _BLK > self.max_cache:
                self._degrade("deg_too_long")
                continue
            if budget is not None:
                need = len(feed) + _BLK - max(0, st.high - len(st.ids))
                if need > budget:
                    self._degrade("deg_no_room")
                    continue
                budget -= need
            try:
                if st.req is None and len(feed) <= _BLK:
                    self._prime(st, feed[:1])
                    feed = feed[1:]
                elif len(feed) > _BLK:
                    self._prime(st, feed[:-_BLK])      # placeholders are never primed (n_mask < _BLK)
                    feed = feed[-_BLK:]
            except _PoolExhausted:
                logger.warning("[InprocVerifier] prime hit an exhausted KV pool; draft skipped")
                self._degrade("deg_prime_pool")
                continue
            if not feed or len(feed) < n_draft:
                self._degrade("deg_empty_feed")
                continue
            parts.append((st, feed))
            meta.append((j, it, len(feed), n_draft, n_mask))
        if not parts:
            return results
        rows_all = []
        i = 0
        try:
            while i < len(parts):
                take = min(len(parts) - i, _LADDER[-1])
                rows_all.extend(self._batched_round(parts[i:i + take]))
                i += take
        except _PoolExhausted:
            lost = len(parts) - len(rows_all)
            self._degrade("deg_round_pool", lost)
            rows_all.extend([None] * lost)
        for (j, it, pl, n_draft, n_mask), (st, _), rows in zip(meta, parts, rows_all):
            if rows is None:
                continue
            ids = rows[pl - n_draft:].argmax(dim=-1).tolist()
            results[j] = [int(x) for x in ids[:n_draft]]
            self.crop_state(st, len(st.ids) - n_mask)   # drop the placeholder slots
        return results

    # ---------------- legacy single-owner shims (bs=1 compatibility) --------------

    def rewind(self, L: int):
        if self.owner is not None:
            st = self._st.get(self.owner)
            if st is not None:
                self.crop_state(st, L)
        self.valid_len = L

    @torch.no_grad()
    def score(self, feed_tokens, need: int, targets, temp: float):
        rpx = self.owner if self.owner is not None else -1
        out = self.batched_score([{"rpx": rpx, "feed": list(feed_tokens),
                                   "need": need, "targets": list(targets),
                                   "temp": temp}])
        st = self._st.get(rpx)
        self.valid_len = len(st.ids) if st else 0
        return out[0]

    # ---------------- init self-test ----------------

    def _self_test(self):
        t0 = time.perf_counter()
        items = [{"rpx": 900 + r, "feed": list(range(1000 + 50 * r, 1100 + 50 * r)),
                  "need": 5, "targets": list(range(1096 + 50 * r, 1101 + 50 * r)),
                  "temp": 1.0} for r in range(3)]
        self.batched_score(items)             # prime + batched tail, bs=3 -> ladder 4
        items2 = [{"rpx": 900 + r, "feed": list(range(2000, 2030)),
                   "need": 14, "targets": list(range(2016, 2030)),
                   "temp": 1.0} for r in range(3)]
        self.batched_score(items2)            # steady batched cycle
        for r in range(3):
            self.crop(900 + r, 80)
        self.batched_score(items2)            # crop + continue
        for r in range(3):
            self.release(900 + r)
        torch.cuda.synchronize()
        if self.stats["degraded"]:
            logger.warning(
                f"[InprocVerifier] SELF-TEST DEGRADED: {self.cum_stats_str()} — the pool "
                f"is too small for even the boot probe; expect the verifier to be inert "
                f"in production (raise inproc_pool_gb).")
        logger.info(f"[InprocVerifier] self-test ok in "
                    f"{(time.perf_counter()-t0)*1e3:.0f}ms")
        self._numeric_self_test()

    @torch.no_grad()
    def _numeric_self_test(self, head: int = 96, tail: int = 16):
        """Does the graph replay actually SEE the prefix?

        The mechanical self-test above only proves nothing crashes. A draft-extend graph
        captured with the wrong ragged/prefix wiring still replays happily -- it just
        scores every position as if the prompt were not there. That failure is invisible
        end-to-end: q silently turns into noise, good draft tokens get rejected, and the
        only symptom is a low accept length that looks like "the model is just weak".

        So: score the SAME tail two ways -- once through the batched graph replay, once
        through plain per-token eager extends -- and compare. They must agree.
        """
        try:
            # THE TEST SEQUENCE MUST BE TEXT THE MODEL IS CONFIDENT ABOUT. A first version
            # used arbitrary token ids; on gibberish the distribution is nearly flat, top-1
            # flips on numerical dust, and the test reported 0.19 agreement on a verifier
            # that is provably fine (reference run, accept 4.67). So build the probe from the model's
            # OWN greedy continuation: every position is high-confidence, and any real
            # corruption then shows up as a large, unambiguous disagreement.
            ste = self._get_st(951)
            seq = [1986, 279, 1156]            # arbitrary short seed
            out = self._prime(ste, seq)
            ref = []
            for _ in range(head + tail):
                nxt = int(out.logits_output.next_token_logits[0].argmax())
                seq.append(nxt)
                out = self._prime(ste, [nxt])  # row sitting ON nxt
                ref.append(out.logits_output.next_token_logits[0].float().clone())
            h, t = seq[:-tail], seq[-tail:]
            e_rows = torch.stack(ref[-tail:])

            stg = self._get_st(950)
            self._prime(stg, h)
            g_rows = self._batched_round([(stg, t)])[0].float()

            # ALIGNMENT: _batched_round returns the logits AT the position of each fed
            # token, so g_rows[j] sits on t[j] -- same row the incremental reference above
            # recorded right after consuming t[j]. (An earlier version compared against the
            # row for h[-1] and scored exactly 0.00, the signature of an off-by-one.)

            # (1) OFFSET SCAN. Comparing two paths invites off-by-one errors in the TEST
            # rather than in the code under test (the first version of this scored exactly
            # 0.00, the signature of a shift). So do not assume the alignment -- search it.
            # If some shift agrees well, the readout is misaligned; if NO shift agrees, the
            # replay is genuinely corrupt.
            best_sh, best_ag = 0, -1.0
            ga, ea = g_rows.argmax(-1), e_rows.argmax(-1)
            for sh in (-2, -1, 0, 1, 2):
                if sh >= 0:
                    a, b = ga[sh:], ea[:len(ea) - sh] if sh else ea
                else:
                    a, b = ga[:sh], ea[-sh:]
                if len(a) == 0:
                    continue
                v = (a == b).float().mean().item()
                if v > best_ag:
                    best_ag, best_sh = v, sh
            # (2) PREFIX SENSITIVITY -- the decisive, alignment-free test. Score the SAME
            # tail after two DIFFERENT prefixes. Any correct attention makes the logits
            # differ; a prefix-blind replay returns bit-identical rows regardless.
            st2 = self._get_st(952)
            h2 = [2000 + (i * 5003) % 30000 for i in range(head)]
            self._prime(st2, h2)
            g2 = self._batched_round([(st2, t)])[0].float()
            same_frac = (g_rows.argmax(-1) == g2.argmax(-1)).float().mean().item()
            # COMPARE PROBABILITIES, NOT LOG-PROBABILITIES. Two earlier criteria failed a
            # verifier that scores 1.00 top-1 agreement: the full-vocab max |dlogp| is set
            # by the far tail (both paths near -30, bf16 dust reads as 12 nats), and even
            # top-8 is still tail once the distribution is peaked (e^-20 vs e^-11 is a huge
            # log gap and a zero-sized probability gap). q is consumed as a PROBABILITY in
            # p_tilde*q >= theta, so the decision-relevant error is the absolute
            # probability gap -- bounded by 1, scale-free, and directly comparable to theta.
            gp = torch.softmax(g_rows, dim=-1)
            ep = torch.softmax(e_rows, dim=-1)
            dmax = (gp - ep).abs().max().item()
            # the number that actually decides an accept: q at the token being scored
            dtop = (gp.gather(1, ea.unsqueeze(1)) - ep.gather(1, ea.unsqueeze(1))
                    ).abs().max().item()
            agree = (ga == ea).float().mean().item()
            logger.info(f"[InprocVerifier] numeric probe: agree@0={agree:.2f} "
                        f"best_agree={best_ag:.2f}@shift{best_sh} "
                        f"prefix_sensitivity: same_top1_under_different_prefix={same_frac:.2f} "
                        f"(1.00 would mean the replay ignores the prefix entirely)")
            # CALIBRATED, NOT ASSUMED. On the reference run (a small proposer where the
            # verifier is independently known good, accept 4.671 in deployment) this probe reads agree 1.00 with
            # max|dp| 0.29: identical argmax everywhere, but bf16 accumulation makes the
            # peak height wander. So 0.29 is NOISE, not breakage, and any tighter bound
            # would condemn a working verifier. The threshold below only catches gross
            # corruption; the real verdict is comparing these numbers against the reference line.
            if agree < 0.9 or dtop > 0.6:
                logger.warning(
                    f"[InprocVerifier] NUMERIC SELF-TEST FAILED: graph replay disagrees "
                    f"with the eager reference (top1 agreement {agree:.2f}, max |dp| "
                    f"{dmax:.4f}, |dp| at the scored token {dtop:.4f}; reference run: "
                    f"1.00 / 0.29). q is unreliable -- accept length will be depressed and "
                    f"theta will look inert. Set inproc_verifier=false to fall back.")
            else:
                logger.info(f"[InprocVerifier] numeric self-test ok "
                            f"(top1 agreement {agree:.2f}, max |dp| {dmax:.4f}, "
                            f"|dp| at the scored token {dtop:.4f}; reference run: 1.00 / 0.29)")
        except Exception as e:
            logger.warning(f"[InprocVerifier] numeric self-test could not run ({e!r})")
        finally:
            for r in (950, 951, 952):
                try:
                    self.release(r)
                except Exception:
                    pass


def _yaml_keys(server_args) -> dict:
    """The algorithm yaml as written, so we can tell 'set to 0.20' from 'defaulted'.

    MarsVerifierStream normalises its config into attributes with defaults filled in, so
    an attribute alone cannot answer "did the operator pin a fraction?". Reading the yaml
    the server was launched with can.
    """
    path = getattr(server_args, "dllm_algorithm_config", None)
    if not path:
        return {}
    try:
        import yaml

        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def maybe_build(target_tp_worker, dllm_algorithm) -> None:
    """Called from tp_worker._init_dllm_algorithm on the TARGET (proposer) worker.

    A failed build stops the server with the reason. The old behaviour — log a WARNING
    and fall back to the transformers verifier chain (_GraphVerifier / _FastVerifier,
    tens of ms per call instead of ~2.6 ms) — made ProVe run far below its paper speed
    with nothing to tell the user. `inproc_fallback: true` in the yaml, or
    MARSV_INPROC_FALLBACK=1, restores that fallback (debugging only).
    """
    path = getattr(dllm_algorithm, "verifier_path", None)
    if not path or not getattr(dllm_algorithm, "use_inproc", True):
        return
    sa = target_tp_worker.server_args
    raw = _yaml_keys(sa)
    allow_fallback = (bool(raw.get("inproc_fallback", False))
                      or os.environ.get("MARSV_INPROC_FALLBACK") == "1")
    try:
        # BACKWARDS COMPATIBILITY: an explicit inproc_mem_fraction in the yaml still
        # pins the raw sglang fraction (the paper's campaign did). Everything else gets
        # the absolute pool target, which is what actually ports across proposer sizes:
        # the same 0.10 fraction was 8029 tokens next to a small proposer and 760 next
        # to the 32B one.
        pinned = raw.get("inproc_mem_fraction", None)
        if pinned is None and not raw and hasattr(dllm_algorithm, "inproc_mem_fraction"):
            # yaml unreadable: fall back to the attribute (old behaviour, never worse)
            pinned = getattr(dllm_algorithm, "inproc_mem_fraction")
        env_pool = os.environ.get("MARSV_INPROC_POOL_GB")
        pool_gb = (float(env_pool) if env_pool else
                   raw.get("inproc_pool_gb",
                           getattr(dllm_algorithm, "inproc_pool_gb", None)))
        if env_pool:
            pinned = None                     # env target wins over a pinned fraction
        conc = getattr(sa, "max_running_requests", None) or _LADDER[-1]
        dllm_algorithm._inproc_verifier = InprocVerifier(
            target_tp_worker, path,
            mem_fraction=(None if pinned is None else float(pinned)),
            max_cache=int(getattr(dllm_algorithm, "inproc_max_cache", 4096)),
            fp8=bool(getattr(dllm_algorithm, "inproc_fp8", False)),
            pool_gb=(None if pool_gb is None else float(pool_gb)),
            reserve_gb=raw.get("inproc_reserve_gb", None),
            expect_concurrency=int(conc))
        dllm_algorithm._inproc_verifier._algo_stats = getattr(
            dllm_algorithm, "_stats", None)
        logger.info("[InprocVerifier] injected into dllm algorithm")
    except Exception as e:
        if not allow_fallback:
            logger.error(f"[InprocVerifier] could not be built: {e!r}")
            raise RuntimeError(
                f"[InprocVerifier] could not be built ({e!r}). ProVe's paper speed needs "
                f"the in-process verifier, so the server refuses to start with the slow "
                f"transformers fallback. Fix the memory split (usually: lower the "
                f"proposer's --mem-fraction-static, or drop a pinned inproc_mem_fraction "
                f"from the yaml so inproc_pool_gb sizes the pool), or set "
                f"`inproc_fallback: true` in the yaml / MARSV_INPROC_FALLBACK=1 to run "
                f"the fallback anyway.") from e
        logger.warning(f"[InprocVerifier] unavailable ({e!r}); algorithm will use "
                       f"its fallback verifier chain (inproc_fallback is set)")
        dllm_algorithm._inproc_verifier = None
