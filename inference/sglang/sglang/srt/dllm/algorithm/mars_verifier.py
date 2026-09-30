# Copyright 2026 The ProVe Authors
# SPDX-License-Identifier: Apache-2.0
"""MarsVerifier — block-fill variant of proposer-verifier decoding, kept for its helpers.

The ProVe configuration shipped and measured in the paper is MarsVerifierStream
(mars_verifier_stream.py); this module is kept because it defines the transformers-based
verifier steppers `_FastVerifier` / `_GraphVerifier` used as ProVe's fallback chain, and
it still registers as `--dllm-algorithm MarsVerifier` (one verifier call per committed
run instead of one per proposed block).

Ordinary speculative decoding: a small model drafts, the big model verifies. ProVe
reverses the roles: the large PROPOSER (a MARS block-trained AR model) already proposes B
tokens per forward (that IS the mask-block decode), and the failure mode of its tau
confidence gate is structural: q, the mask-slot confidence, is computed BLIND — mask
placeholders hide the actually-realized left neighbours, so q cannot see cross-position
inconsistency, which is the one inherent degradation of multi-token prediction. A smaller
AR VERIFIER (a stock Qwen2.5-3B/7B-Instruct in the paper) reads the REALIZED chain and
supplies exactly the missing information.

Why a small verifier can judge a large proposer: this is a DISCRIMINATOR task, not a
generator task. The verifier never needs to know the right answer — the proposal came
out of the proposer; the verifier only checks that this proposed token is consistent
with the left context actually sampled. Discrimination is an order of magnitude easier
than generation.

ACCEPT RULE  (the three deliberate choices, per design):
  1. ratio rule:   accept x at slot j  iff  p~(x) >= theta * q(x)
       p~ = softmax(verifier_logits / verifier_temp)[x]  — the verifier, conditioned on
            the realized chain (prompt + committed + earlier proposed tokens in this block)
       q  = softmax(proposer mask-slot logits)[x]        — the proposal's own probability
     NOT "verifier top-1 == proposed token" (that would project decoding onto the
     verifier's argmax — you'd be decoding the small model). NOT a raw p~ threshold
     (high-entropy positions would be systematically rejected; dividing by q normalizes
     the entropy away). ProVe (MarsVerifierStream) uses the product rule p~ * q >= theta
     instead; see the paper.
  2. first token free: the leftmost mask's logit comes (right-shift convention) from a
     REAL committed token, so it is the exact clean-AR distribution — always committed,
     bypassing the verifier. Worst case degrades to vanilla AR + small overhead.
  3. free audit: the block-final commit forward re-computes logits with every committed
     token as a real input. Reading them gives ground-truth "would AR have said this?"
     for every emitted token at zero extra cost -> false-accept monitoring (audit: true).
     (Rollback/lossless mode would build on the same signal; not in v0.)

Baseline to beat: the tau confidence rule (MarsThreshold), NOT plain AR. The verifier only
deserves its VRAM if it Pareto-dominates tau on span-vs-quality. The advantage should GROW
with block size B (deeper masks -> more cross-position mismatch -> blinder q).

Config keys (--dllm-algorithm-config YAML), on top of MarsThreshold's:
  verifier_path:  HF id or local path of the small AR verifier (same tokenizer family!).
  theta:          ratio-rule threshold in (0, 1]. Higher = stricter. Default 0.6.
  verifier_temp:  temperature calibration for p~. Default 1.0.
  audit:          bool, default true — read the free audit signal into stats.
The variables named `drafts` below hold the proposed block.
"""
import logging
import os
import time
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class _FastVerifier:
    """Compiled fixed-shape verifier stepper (transformers): chunk-16 + StaticCache + rewind-as-crop.

    Eager HF is launch-bound: 87 ms per 16-token chunk (measured standalone). This path
    is torch.compile(mode="reduce-overhead", fullgraph) over ONE fixed shape -> a CUDA
    graph replay at ~5.1 ms/step (17x). Prompt prefill reuses the same step (chunk-16
    loop, one-time ~15 calls/request). "Crop" is free: with position-based causal
    masking, rewinding cache_position excludes stale KV automatically.

    SINGLE-SEQUENCE by design: owns one StaticCache; the owner request switches with a
    full re-feed (fine for the sequential bench; batched serving keeps the eager
    DynamicCache fallback).
    """

    def __init__(self, model, chunk: int = 16, max_cache: int = 4096):
        self.m = model
        self.chunk = chunk
        self.max_cache = max_cache
        dev = next(model.parameters()).device
        # Inductor's parallel compile forks worker processes; forking a CUDA-initialized
        # process (the sglang scheduler) deadlocks. Single-thread the compile and keep
        # its cache on LOCAL disk (never a FUSE mount).
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/cache/torchinductor_marsv")
        import torch._inductor.config as _icfg
        _icfg.compile_threads = 1
        # Inductor's cudagraph-trees SEGFAULT on first capture inside the sglang
        # scheduler (conflict with sglang's own captured-graph memory pools — verified
        # by stack: cudagraph_trees.add_function -> static_cuda_launcher.run -> SIGSEGV).
        # Keep the kernel fusion, drop inductor's graphs. (Endgame if more speed is
        # needed: MANUAL torch.cuda.graph capture like sglang does its own.)
        _icfg.triton.cudagraphs = False
        from transformers import StaticCache
        self.cache = StaticCache(config=model.config, max_batch_size=1,
                                 max_cache_len=max_cache, device=dev,
                                 dtype=torch.bfloat16)
        self.ids = torch.zeros((1, chunk), dtype=torch.long, device=dev)
        self.pos = torch.zeros((1, chunk), dtype=torch.long, device=dev)
        self.cpos = torch.zeros((chunk,), dtype=torch.long, device=dev)
        self.cm = torch.compile(model, fullgraph=True)   # fused kernels, no cudagraphs
        self.valid_len = 0
        self.owner = None

    def rewind(self, L: int):
        self.valid_len = L

    @torch.no_grad()
    def feed(self, tokens, need_last_n: int):
        """Append tokens at valid_len; return logits of the LAST need_last_n fed
        positions (fp32, detached copies — the graph reuses its output buffers)."""
        dev = self.ids.device
        C = self.chunk
        keep = []
        i = 0
        n = len(tokens)
        while i < n:
            part = tokens[i:i + C]
            pl = len(part)
            L = self.valid_len
            self.ids.zero_()
            self.ids[0, :pl] = torch.tensor(part, dtype=torch.long, device=dev)
            self.pos.copy_(torch.arange(L, L + C, device=dev).unsqueeze(0))
            self.cpos.copy_(torch.arange(L, L + C, device=dev))
            torch.compiler.cudagraph_mark_step_begin()
            lg = self.cm(input_ids=self.ids, position_ids=self.pos,
                         past_key_values=self.cache, cache_position=self.cpos,
                         use_cache=True).logits
            # only clone rows we might need (fed real positions of this chunk)
            fed_start = i
            fed_end = i + pl
            if fed_end > n - need_last_n:
                lo = max(0, (n - need_last_n) - fed_start)
                keep.append(lg[0, lo:pl].float().clone())
            self.valid_len = L + pl   # padded tail is beyond valid_len -> never attended
            i += pl
        return torch.cat(keep, dim=0) if keep else None


class _GraphVerifier:
    """Manually captured CUDA-graph chunk-16 stepper (same interface as _FastVerifier).

    Inductor's cudagraph-trees segfault inside the sglang scheduler (pool conflict with
    sglang's own captured graphs), so this captures the EAGER chunk step with
    torch.cuda.CUDAGraph in a private memory pool instead. Capture is lazy — first feed()
    happens at serving time, long after sglang's own boot-time captures. All chunk state
    (ids/pos/cpos/StaticCache/logits) is static; per step we copy inputs in and replay.
    Pad-tail KV slots hold garbage but are overwritten by the next chunk before any query
    can attend to them (same argument as _FastVerifier).
    """

    def __init__(self, model, chunk: int = 32, max_cache: int = 4096):
        # chunk 32: one replay covers a whole stream cycle (k+15 <= 31 tokens) —
        # halves ms/cycle vs chunk 16 (8.6 vs 17.0; sdpa cost is KV-length-bound).
        # max_cache MUST stay 4096: 2048 segfaults on replay (cudnn sdpa kernel
        # selection differs; empirically bisected). Compiled(triton)-kernels-inside-
        # manual-graph also segfault on replay (torch 2.9) -> the captured step is EAGER.
        self.m = model
        self.chunk = chunk
        self.max_cache = max_cache
        dev = next(model.parameters()).device
        from transformers import StaticCache
        self.cache = StaticCache(config=model.config, max_batch_size=1,
                                 max_cache_len=max_cache, device=dev,
                                 dtype=torch.bfloat16)
        self.ids = torch.zeros((1, chunk), dtype=torch.long, device=dev)
        self.pos = torch.zeros((1, chunk), dtype=torch.long, device=dev)
        self.cpos = torch.zeros((chunk,), dtype=torch.long, device=dev)
        self.graph = None
        self.static_logits = None
        self.valid_len = 0
        self.owner = None

    def rewind(self, L: int):
        self.valid_len = L

    def _capture(self):
        """3 warmup steps on a side stream, then capture one chunk step."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.m(input_ids=self.ids, position_ids=self.pos,
                       past_key_values=self.cache, cache_position=self.cpos,
                       use_cache=True)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.m(input_ids=self.ids, position_ids=self.pos,
                         past_key_values=self.cache, cache_position=self.cpos,
                         use_cache=True)
            self.static_logits = out.logits
        self.graph = g
        # warmup wrote garbage into cache slots 0..chunk; the first real chunk
        # overwrites them (cpos starts at valid_len=0), so no reset needed.

    @torch.no_grad()
    def feed(self, tokens, need_last_n: int):
        """Same contract as _FastVerifier.feed."""
        if self.graph is None:
            self._capture()
        dev = self.ids.device
        C = self.chunk
        keep = []
        i = 0
        n = len(tokens)
        while i < n:
            part = tokens[i:i + C]
            pl = len(part)
            L = self.valid_len
            self.ids.zero_()
            self.ids[0, :pl] = torch.as_tensor(part, dtype=torch.long).to(dev, non_blocking=True)
            torch.arange(L, L + C, out=self.cpos)
            self.pos.copy_(self.cpos.unsqueeze(0))
            self.graph.replay()
            fed_start = i
            fed_end = i + pl
            if fed_end > n - need_last_n:
                lo = max(0, (n - need_last_n) - fed_start)
                keep.append(self.static_logits[0, lo:pl].float().clone())
            self.valid_len = L + pl
            i += pl
        return torch.cat(keep, dim=0) if keep else None


class MarsVerifier(DllmAlgorithm):

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)  # unused by gate
        self.theta = float(config.algorithm_config.get("theta", 0.6))
        # Accept rule. Offline arbitration on 36k deploy-population points (see
        # verifier_analysis.py) OVERTURNED the ratio rule:
        #   tau (q alone)        AUC 0.9281
        #   ratio p~/q           AUC 0.8816   <- dividing by q SUBTRACTS the proposer's
        #                                        information; verifier noise dominates
        #                                        exactly where q is most confident
        #   raw p~               AUC 0.9566
        #   p~ * q  (AND)        AUC 0.9629   <- naive-Bayes fusion of two witnesses
        # min(1, p/q) is the right rule for UNBIASED SAMPLING (spec-dec's rejection
        # step); this gate is a RISK CLASSIFIER, and for classification the evidence
        # multiplies. "and": accept iff p~ * q >= theta. "ratio": p~ >= theta * q.
        self.rule = config.algorithm_config.get("rule", "and")
        # compiled fast path (chunk-16 CUDA-graph stepper); eager DynamicCache fallback
        self.use_fast = bool(config.algorithm_config.get("fast_verifier", True))
        self._fv = None
        self.verifier_path = config.algorithm_config.get(
            "verifier_path", "Qwen/Qwen2.5-3B-Instruct"
        )
        self.verifier_temp = float(config.algorithm_config.get("verifier_temp", 1.0))
        self.audit = bool(config.algorithm_config.get("audit", True))
        self.temperature = float(config.algorithm_config.get("temperature", 0.0))
        # structure-aware force-AR (parity with MarsThreshold; default off)
        self.code_aware_tau = config.algorithm_config.get("code_aware_tau", False)
        self.code_open_token = int(config.algorithm_config.get("code_open_token", 151657))
        self.code_close_token = int(config.algorithm_config.get("code_close_token", 151658))
        self._in_code = {}

        self._verifier = None            # lazy-loaded on first decode
        self._prefix_ids = {}            # rpx -> list[int], realized sequence
        self._v_cache = {}               # rpx -> transformers DynamicCache
        self._v_len = {}                 # rpx -> tokens covered by the cache
        # Pending protocol (same fix as MarsThreshold): the pure-decode template hands
        # blocks over ALL-MASK; the algorithm must seed block[0] with the next token
        # (exact AR, from the previous commit/prefill forward's last-row logits).
        self._pending = {}
        for k in ("verifier_calls", "verifier_time_ms", "v_accept", "v_reject",
                  "audit_pairs", "audit_mismatch", "audit_mismatch_vslots",
                  "total_forwards", "total_tokens"):
            self._stats.setdefault(k, 0)

    # ---------------- verifier plumbing ----------------

    def _load_verifier(self, device):
        if self._verifier is not None:
            return
        from transformers import AutoModelForCausalLM
        t0 = time.perf_counter()
        self._verifier = AutoModelForCausalLM.from_pretrained(
            self.verifier_path, dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
        for p in self._verifier.parameters():
            p.requires_grad_(False)
        if self.use_fast:
            try:
                self._fv = _FastVerifier(self._verifier)
                logger.info("[MarsVerifier] fast (compiled) stepper enabled — first "
                            "scoring call pays ~40s of torch.compile")
            except Exception as e:
                logger.warning(f"[MarsVerifier] fast stepper unavailable ({e!r}); "
                               "falling back to eager DynamicCache")
                self._fv = None
        logger.info(
            f"[MarsVerifier] loaded {self.verifier_path} in "
            f"{time.perf_counter()-t0:.1f}s "
            f"({sum(p.numel() for p in self._verifier.parameters())/1e9:.2f}B params), "
            f"theta={self.theta} rule={self.rule} v_temp={self.verifier_temp}"
        )

    def cleanup_request(self, req_pool_idx: int):
        self._in_code.pop(req_pool_idx, None)
        self._prefix_ids.pop(req_pool_idx, None)
        self._v_cache.pop(req_pool_idx, None)
        self._v_len.pop(req_pool_idx, None)
        self._pending.pop(req_pool_idx, None)
        if self._fv is not None and self._fv.owner == req_pool_idx:
            self._fv.owner = None
            self._fv.rewind(0)
        if os.environ.get("MARSV_LOG"):
            s = self._stats
            ap, am = s["audit_pairs"], s["audit_mismatch"]
            logger.info(
                f"[MarsVerifier] cum stats: fwd={s['total_forwards']} tok={s['total_tokens']} "
                f"tok/fwd={s['total_tokens']/max(1,s['total_forwards']):.3f} "
                f"v_accept={s['v_accept']} v_reject={s['v_reject']} "
                f"audit_false_accept={am}/{ap} ({am/max(1,ap):.4f}) "
                f"vslot_mismatch={s['audit_mismatch_vslots']} "
                f"v_time={s['verifier_time_ms']:.0f}ms/{s['verifier_calls']}calls"
            )

    def _sync_prefix(self, rpx, entry_tokens: List[int]):
        """Align our realized-sequence record with the block's committed entry region.

        The scheduler seeds each new block with the previously committed token(s); most are
        already in _prefix_ids (we append every emitted slice), but e.g. the token sampled
        at prefill time arrives here first. Maximal-suffix-align and append what's missing.
        """
        pref = self._prefix_ids.setdefault(rpx, [])
        n = len(entry_tokens)
        for o in range(min(n, len(pref)), -1, -1):
            if o == 0 or pref[-o:] == entry_tokens[:o]:
                pref.extend(entry_tokens[o:])
                return

    @torch.no_grad()
    def _score_chain(self, rpx, realized: List[int], drafts: List[int], device):
        """p~ for the proposed tokens drafts[1:], each conditioned on realized + drafts[:i] (the accepted-path
        distribution). One verifier forward: feed uncached-realized + drafts[:-1]; the
        logits at the last len(drafts)-1 positions predict drafts[1:]."""
        from transformers import DynamicCache
        self._load_verifier(device)
        t0 = time.perf_counter()
        if self._fv is not None:
            fv = self._fv
            if fv.owner != rpx:
                # owner switch: re-feed this request's realized prefix from scratch
                # (one-time chunk-16 loop; sequential serving switches rarely)
                fv.owner = rpx
                fv.rewind(0)
                self._v_len[rpx] = 0
            vlen = self._v_len.get(rpx, 0)
            feed = realized[vlen:] + drafts[:-1]
            n_score = len(drafts) - 1
            if fv.valid_len + len(feed) + fv.chunk > fv.max_cache:
                self._fv_overflow = True   # too long for the static cache: skip scoring
                return torch.zeros(n_score, device=device)
            lg = fv.feed(feed, n_score)
            if self.verifier_temp != 1.0:
                lg = lg / self.verifier_temp
            lp = torch.log_softmax(lg, dim=-1)
            tgt = torch.tensor(drafts[1:], dtype=torch.long, device=lg.device)
            p_tilde = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).exp()
            self._v_len[rpx] = fv.valid_len
            self._stats["verifier_calls"] += 1
            self._stats["verifier_time_ms"] += (time.perf_counter() - t0) * 1e3
            return p_tilde
        cache = self._v_cache.get(rpx)
        if cache is None:
            cache = DynamicCache()
            self._v_cache[rpx] = cache
            self._v_len[rpx] = 0
        vlen = self._v_len[rpx]
        feed = realized[vlen:] + drafts[:-1]
        n_score = len(drafts) - 1
        ids = torch.tensor([feed], dtype=torch.long, device=device)
        pos = torch.arange(vlen, vlen + len(feed), device=device).unsqueeze(0)
        out = self._verifier(input_ids=ids, position_ids=pos,
                             past_key_values=cache, use_cache=True)
        lg = out.logits[0, -n_score:].float()
        if self.verifier_temp != 1.0:
            lg = lg / self.verifier_temp
        lp = torch.log_softmax(lg, dim=-1)
        tgt = torch.tensor(drafts[1:], dtype=torch.long, device=device)
        p_tilde = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).exp()   # [n_score]
        # cache now covers realized + drafts[:-1]; caller crops after acceptance
        self._v_len[rpx] = vlen + len(feed)
        self._stats["verifier_calls"] += 1
        self._stats["verifier_time_ms"] += (time.perf_counter() - t0) * 1e3
        return p_tilde

    def _crop_cache(self, rpx, keep_len: int):
        if self._fv is not None:
            if self._fv.owner == rpx and self._v_len.get(rpx, 0) > keep_len:
                self._fv.rewind(keep_len)     # free: position-masked, stale KV ignored
                self._v_len[rpx] = keep_len
            return
        cache = self._v_cache.get(rpx)
        if cache is None:
            return
        if self._v_len.get(rpx, 0) > keep_len:
            cache.crop(keep_len)
            self._v_len[rpx] = keep_len

    # ---------------- misc (ported from MarsThreshold) ----------------

    def _rpx(self, forward_batch, bid: int):
        try:
            return int(forward_batch.req_pool_indices[bid].item())
        except Exception:
            return None

    def _update_in_code(self, rpx, tokens):
        if not self.code_aware_tau or rpx is None:
            return
        for t in tokens:
            t = int(t)
            if t == self.code_open_token:
                self._in_code[rpx] = True
            elif t == self.code_close_token:
                self._in_code[rpx] = False

    # ---------------- main ----------------

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        num_forwards = 0
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: prefill / no masks. Capture the prompt tokens per request (the
        # verifier conditions on them) AND the next-token seed from the chunk's last
        # row (chunked prefill overwrites it; the final chunk wins).
        if torch.sum(mask_index).item() == 0:
            _el = forward_batch.extend_seq_lens
            lens = None
            if _el is not None:
                lens = _el.tolist() if isinstance(_el, torch.Tensor) else list(_el)
                ids_cpu = forward_batch.input_ids.tolist()
                off = 0
                for bid, ln in enumerate(lens):
                    rpx = self._rpx(forward_batch, bid)
                    if rpx is not None:
                        self._prefix_ids.setdefault(rpx, []).extend(
                            ids_cpu[off:off + ln])
                    off += ln
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            self._stats["total_forwards"] += 1
            fl = getattr(out.logits_output, "full_logits", None)
            if fl is not None and lens is not None:
                off = 0
                for bid, ln in enumerate(lens):
                    rpx = self._rpx(forward_batch, bid)
                    if rpx is not None and ln > 0:
                        self._pending[rpx] = int(
                            torch.argmax(fl[off + ln - 1]).item())
                    off += ln
            return out.logits_output, [], out.can_run_graph

        start_list = []
        ws_per_block = [[0] * self.block_size for _ in range(batch_size)]
        # positions accepted BY THE VERIFIER (not leftmost-free), for the audit split
        vslots_per_block = [set() for _ in range(batch_size)]

        for block_id in range(batch_size):
            bs_ = block_id * self.block_size
            be_ = bs_ + self.block_size
            block_input_ids = forward_batch.input_ids[bs_:be_]
            start = self.block_size - int(
                torch.sum(block_input_ids == self.mask_id).item())
            rpx = self._rpx(forward_batch, block_id)
            if start == 0:
                # pure-decode template: seed block[0] with the pending exact-AR token;
                # start stays 0 so it is emitted this round.
                pend = self._pending.pop(rpx, None) if rpx is not None else None
                if pend is not None:
                    block_input_ids[0] = pend
                    if rpx is not None:
                        self._prefix_ids.setdefault(rpx, []).append(pend)
                        self._update_in_code(rpx, [pend])
            else:
                if rpx is not None:
                    self._pending.pop(rpx, None)
                    entry = block_input_ids[:start].tolist()
                    self._sync_prefix(rpx, entry)
                    self._update_in_code(rpx, entry)
            start_list.append(start)

        # Inner loop: each proposer forward commits a left-to-right run per block.
        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            num_forwards += 1

            for bid in range(batch_size):
                bs_ = bid * self.block_size
                be_ = bs_ + self.block_size
                block_input_ids = forward_batch.input_ids[bs_:be_]
                block_mask = block_input_ids == self.mask_id
                n_masks = int(torch.sum(block_mask).item())
                if n_masks == 0:
                    continue
                curr_logits = logits_output.full_logits[bs_:be_]
                # right-shift convention: logit at p-1 predicts token p
                curr_logits = torch.roll(curr_logits, shifts=1, dims=0)
                probs = F.softmax(curr_logits, dim=-1)
                if self.temperature > 0:
                    x = torch.multinomial(
                        F.softmax(curr_logits / self.temperature, dim=-1), 1
                    ).squeeze(-1)
                else:
                    x = torch.argmax(curr_logits, dim=-1)
                # q of the EMITTED token (not the top-1): the ratio rule needs q(x)
                q = probs.gather(1, x.unsqueeze(1)).squeeze(1)

                mask_pos = block_mask.nonzero(as_tuple=True)[0].tolist()
                drafts = [int(x[p].item()) for p in mask_pos]
                qs = [float(q[p].item()) for p in mask_pos]
                rpx = self._rpx(forward_batch, bid)

                # committed region of the block = realized tail; verifier conditions on it
                realized = self._prefix_ids.get(rpx, []) if rpx is not None else []

                accept_n = 1  # choice 2: leftmost mask = exact clean AR, free commit
                force_ar = (self.code_aware_tau and rpx is not None
                            and self._in_code.get(rpx, False))
                if n_masks > 1 and not force_ar and rpx is not None:
                    p_tilde = self._score_chain(rpx, realized, drafts, device)
                    for i in range(n_masks - 1):
                        if self.rule == "and":
                            ok = float(p_tilde[i]) * qs[i + 1] >= self.theta
                        else:  # "ratio" (kept for A/B)
                            ok = float(p_tilde[i]) >= self.theta * qs[i + 1]
                        if ok:
                            accept_n += 1
                            self._stats["v_accept"] += 1
                        else:
                            self._stats["v_reject"] += 1
                            break
                    # keep only accepted proposed tokens in the verifier cache
                    self._crop_cache(rpx, len(realized) + min(accept_n, n_masks - 1))

                acc_pos = mask_pos[:accept_n]
                ws_per_block[bid][acc_pos[0]] = 1
                for k, p in enumerate(acc_pos):
                    block_input_ids[p] = drafts[k]
                    if k > 0:
                        vslots_per_block[bid].add(p)
                if rpx is not None:
                    self._prefix_ids[rpx].extend(drafts[:accept_n])
                    self._update_in_code(rpx, drafts[:accept_n])

        # Commit forward: refresh KV with finalized tokens. Free audit source (choice 3):
        # every input is now a real token, so these logits are exact AR. The last row is
        # also the exact-AR prediction for the NEXT block's seed.
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        num_forwards += 1
        _fl = getattr(logits_output, "full_logits", None)
        if _fl is not None:
            for bid in range(batch_size):
                rpx = self._rpx(forward_batch, bid)
                if rpx is not None:
                    self._pending[rpx] = int(torch.argmax(
                        _fl[bid * self.block_size + self.block_size - 1]).item())

        if self.audit:
            for bid in range(batch_size):
                bs_ = bid * self.block_size
                be_ = bs_ + self.block_size
                blk = forward_batch.input_ids[bs_:be_]
                lg = torch.roll(
                    logits_output.full_logits[bs_:be_], shifts=1, dims=0)
                ar_top1 = torch.argmax(lg, dim=-1)
                for p in range(max(1, start_list[bid]), self.block_size):
                    self._stats["audit_pairs"] += 1
                    if int(ar_top1[p].item()) != int(blk[p].item()):
                        self._stats["audit_mismatch"] += 1
                        if p in vslots_per_block[bid]:
                            self._stats["audit_mismatch_vslots"] += 1

        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i]:] for i in range(batch_size)]
        total_new = sum(self.block_size - s for s in start_list)
        self._stats["total_forwards"] += num_forwards
        self._stats["total_tokens"] += total_new
        logits_output.customized_info = {
            "mars_ws": [ws_per_block[i][start_list[i]:] for i in range(batch_size)]
        }
        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = MarsVerifier
