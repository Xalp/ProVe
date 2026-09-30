# Copyright 2026 The ProVe Authors
# SPDX-License-Identifier: Apache-2.0
"""MarsVerifierStream — ProVe: the proposer proposes a block, a smaller verifier judges it.

ONE proposer forward per cycle: the proposer's mask stream proposes a block of tokens in
one forward (argmax per slot), the verifier (a smaller stock AR model, served in-process
by inproc_verifier.py) scores the WHOLE proposed block against the realized context in
one more forward, the accepted prefix commits, and the next proposer forward re-anchors
after it. No inner re-proposal loop (MarsVerifier/MarsThreshold's block-fill), no
self-verification rows (IDLMBlockN's 15 spec rows), no second proposer forward ever.

Why this shape wins twice:
  DEPTH.  The block-fill loop re-proposes after every short commit run, so gate decisions
          concentrate at depths 1-3 — where the verifier's measured edge is ~0. Here every
          cycle judges the full proposed block, so the deep slots (offline edge +.03..+.044,
          9x the shallow edge) are finally exercised.
  ROWS.   IDLM pays 31 rows/forward because exact self-verification needs the 15 spec rows
          re-fed. Outsourcing verification to the small verifier cuts the proposer to a
          16-row window: [k committed re-fed][16-k masks]. tok/row roughly doubles IDLM —
          the metric that wins the compute-bound (high-concurrency) regime.

Window algebra (all verified against the scheduler contract):
  window at S..S+15 = [c1..ck real re-fed][m x (16-k)],  prefix KV = S
  row(c_i) -> exact-AR audit of c_{i+1}; row(ck) -> y1 (EXACT AR, committed free)
  row(m_i) -> y_{i+1} mask-conditioned proposed token, judged by the verifier's chain score
  accept y1..yj  ->  emit j tokens; advance = k; trim = 16-k mask rows (window tail);
  kv_committed lands exactly at next window start S+k. Steady state j = k.
  Worst case degrades to vanilla AR (y1 alone), never below.

FREE AUDIT (no commit forward exists, but none is needed): this round's re-fed rows give
exact-AR predictions for c2..ck — precisely the tokens the verifier accepted last round.
False accepts are counted per token at zero cost; a rollback/lossless mode would hook here.

Config keys: verifier_path, theta, rule ("and" | "ptilde"), theta_deep + deep_start
(optional two-stage: rule for depth < deep_start, p~-only >= theta_deep beyond),
verifier_temp, audit, fast_verifier, inproc_pool_gb / inproc_mem_fraction (verifier
memory; see inproc_verifier.py). Greedy only (temperature ignored; the proposed block is
the argmax of every slot by design). The variables named `drafts` below hold the
proposed block.
"""
import logging
import os
import time
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.mars_verifier import _FastVerifier, _GraphVerifier
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class MarsVerifierStream(DllmAlgorithm):

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.theta = float(config.algorithm_config.get("theta", 0.45))
        # hybrid gate: proposed tokens with q >= q_hi auto-accept without consulting the
        # verifier; the verifier is only called on cycles whose acceptance walk hits an
        # ambiguous token. q_hi=1.0 disables (pure verifier gate). 0.95 = the old
        # tau operating point (audit FA 0.16% there).
        self.q_hi = float(config.algorithm_config.get("q_hi", 1.0))
        # vtok: at the position where the gate stops, also commit the VERIFIER's own
        # argmax token, so a rejected cycle still gains one token the way the exact
        # configurations gain their correction token. Lossy by construction (the token
        # comes from the small model), so it is off unless the yaml asks for it.
        _vt = config.algorithm_config.get("vtok", False)
        # "v"  = commit the verifier's own argmax at the stop position
        # "pq" = commit the candidate that maximizes the GATE score p_tilde * q over the
        #        verifier's top-K there, so the proposer keeps a vote (recommended)
        self.vtok = ("v" if _vt is True else (str(_vt).lower() if _vt else ""))
        if self.vtok not in ("", "v", "pq"):
            self.vtok = "v" if self.vtok in ("true", "1", "yes") else ""
        self.rule = config.algorithm_config.get("rule", "and")
        # joint: decode every judged position by the candidate that maximizes
        # p_tilde(v) * q(v) over the verifier's top-K, instead of trusting the proposer's
        # argmax; theta guards depth (stop when the best product is below it), and a
        # candidate that differs from the proposed token is committed as a correction and
        # stops the cycle (the deeper proposed tokens were conditioned on the old token). y1 stays the
        # proposer's exact AR token. Quality experiment (2026-09-21), off by default.
        self.joint = bool(config.algorithm_config.get("joint", False))
        self.deep_start = int(config.algorithm_config.get("deep_start", 0))  # 0 = off
        self.theta_deep = float(config.algorithm_config.get("theta_deep", 0.3))
        self.verifier_path = config.algorithm_config.get("verifier_path", None)
        if (not self.verifier_path
                or str(self.verifier_path).strip() in ("", "__VERIFIER__")):
            raise ValueError(
                "ProVe / MarsVerifierStream needs a verifier model: set "
                "`verifier_path: <local dir of a stock AR model, e.g. a download of "
                "Qwen/Qwen2.5-3B-Instruct>` in the yaml passed with "
                "--dllm-algorithm-config. The shipped configs carry the placeholder "
                "__VERIFIER__, which `serve.sh --verifier <dir>` fills in; replace it "
                "by hand if you launch sglang yourself (see inference/configs/README.md). "
                "For the lossless configuration, which needs no verifier, use "
                "--dllm-algorithm ProVeLossless instead.")
        self.verifier_temp = float(config.algorithm_config.get("verifier_temp", 1.0))
        self.audit = bool(config.algorithm_config.get("audit", True))
        self.use_fast = bool(config.algorithm_config.get("fast_verifier", True))
        self.use_graph = bool(config.algorithm_config.get("graph_verifier", True))
        if config.algorithm_config.get("sgl_verifier", False):
            raise ValueError(
                "`sgl_verifier: true` selects a subprocess verifier that is not part of "
                "this release (its numerics were never validated); remove the key. The "
                "in-process verifier (`inproc_verifier: true`, the default) is the "
                "configuration used in the paper.")
        self.use_inproc = bool(config.algorithm_config.get("inproc_verifier", True))
        self.inproc_fp8 = bool(config.algorithm_config.get("inproc_fp8", False))
        # Verifier memory. The in-process verifier sizes its own KV pool from
        # `inproc_pool_gb` (absolute GB, default 2.0 — portable across proposer and GPU
        # sizes). An explicit `inproc_mem_fraction` instead PINS sglang's raw fraction,
        # which is a share of the memory the proposer left over, minus the verifier's
        # weights (pool = free_after_proposer * fraction - weights; inproc_verifier.py).
        # The paper pinned 0.18 (3B) / 0.32 (7B) next to the 32B proposer served with
        # --mem-fraction-static 0.70 / 0.55 at TP=2. Both keys are read from the yaml by
        # inproc_verifier.maybe_build; they are mirrored here for introspection only.
        _imf = config.algorithm_config.get("inproc_mem_fraction", None)
        self.inproc_mem_fraction = (float(_imf) if _imf is not None else None)
        _ipg = config.algorithm_config.get("inproc_pool_gb", None)
        self.inproc_pool_gb = (float(_ipg) if _ipg is not None else None)
        self.inproc_max_cache = int(
            config.algorithm_config.get("inproc_max_cache", 4096))
        # DSpark-style sequential head: rank-256 bigram bias on mask-slot logits,
        # applied left-to-right over the proposed block with the exact top-K partition
        # correction. Path to {"W1","W2"} ckpt enables it.
        self.seq_head_path = config.algorithm_config.get("seq_head", None)
        self._sh_W1 = self._sh_W2 = None

        self._verifier = None
        self._fv = None
        self._prefix_ids = {}      # rpx -> realized (committed) ids incl. prompt
        self._v_cache = {}         # eager fallback DynamicCache per rpx
        self._v_len = {}
        self._refeed = {}          # rpx -> tokens accepted last round, to re-feed
        self._emit_refeed = {}     # rpx -> bool, True only for the first-round t0
        self._pending = {}         # rpx -> t0 from prefill logits
        for k in ("verifier_calls", "verifier_time_ms", "v_accept", "v_reject",
                  "audit_pairs", "audit_mismatch",
                  "total_forwards", "total_tokens"):
            self._stats.setdefault(k, 0)

    # ---------------- verifier plumbing (shared semantics with MarsVerifier) ----------

    def _load_verifier(self, device):
        if self._fv is not None or self._verifier is not None:
            return
        t0 = time.perf_counter()
        # injected at tp_worker._init_dllm_algorithm (in-process EAGLE-style runner)
        iv = getattr(self, "_inproc_verifier", None)
        if iv is not None:
            self._fv = iv
            logger.info("[MarsVerifierStream] using in-process EAGLE-style verifier")
            return
        from transformers import AutoModelForCausalLM
        self._verifier = AutoModelForCausalLM.from_pretrained(
            self.verifier_path, dtype=torch.bfloat16,
            attn_implementation="sdpa").to(device).eval()
        for p in self._verifier.parameters():
            p.requires_grad_(False)
        if self.use_graph:
            try:
                self._fv = _GraphVerifier(self._verifier)
                logger.info("[MarsVerifierStream] manual-graph stepper enabled")
            except Exception as e:
                logger.warning(f"[MarsVerifierStream] graph stepper unavailable ({e!r})")
                self._fv = None
        if self._fv is None and self.use_fast:
            try:
                self._fv = _FastVerifier(self._verifier)
                logger.info("[MarsVerifierStream] fast stepper enabled")
            except Exception as e:
                logger.warning(f"[MarsVerifierStream] fast stepper unavailable ({e!r})")
                self._fv = None
        logger.info(f"[MarsVerifierStream] loaded {self.verifier_path} "
                    f"in {time.perf_counter()-t0:.1f}s theta={self.theta} "
                    f"rule={self.rule} deep_start={self.deep_start}")

    def cleanup_request(self, req_pool_idx: int):
        for d in (self._prefix_ids, self._v_cache, self._v_len,
                  self._refeed, self._emit_refeed, self._pending):
            d.pop(req_pool_idx, None)
        if self._fv is not None and hasattr(self._fv, "release"):
            self._fv.release(req_pool_idx)
        elif self._fv is not None and self._fv.owner == req_pool_idx:
            self._fv.owner = None
            self._fv.rewind(0)
        if os.environ.get("MARSV_LOG"):
            s = self._stats
            ap, am = s["audit_pairs"], s["audit_mismatch"]
            logger.info(
                f"[MarsVerifierStream] cum stats: fwd={s['total_forwards']} "
                f"tok={s['total_tokens']} "
                f"tok/fwd={s['total_tokens']/max(1,s['total_forwards']):.3f} "
                f"v_accept={s['v_accept']} v_reject={s['v_reject']} "
                f"audit_false_accept={am}/{ap} ({am/max(1,ap):.4f}) "
                f"v_time={s['verifier_time_ms']:.0f}ms/{s['verifier_calls']}calls")

    @torch.no_grad()
    def _score_chain(self, rpx, realized: List[int], drafts: List[int], device):
        """p~ for the proposed tokens drafts[1:], each conditioned on realized + drafts[:i]; one verifier pass."""
        self._load_verifier(device)
        t0 = time.perf_counter()
        n_score = len(drafts) - 1
        if self._fv is not None:
            fv = self._fv
            if fv.owner != rpx:
                fv.owner = rpx
                fv.rewind(0)
                self._v_len[rpx] = 0
            vlen = self._v_len.get(rpx, 0)
            feed = realized[vlen:] + drafts[:-1]
            if fv.valid_len + len(feed) + fv.chunk > fv.max_cache:
                return None                        # too long: caller falls back to y1-only
            if hasattr(fv, "score"):
                # worker-side scoring: one ipc hop, p~ floats back (no logits transfer)
                p_tilde = fv.score(feed, n_score, drafts[1:], self.verifier_temp)
                self._v_len[rpx] = fv.valid_len
                self._stats["verifier_calls"] += 1
                self._stats["verifier_time_ms"] += (time.perf_counter() - t0) * 1e3
                return p_tilde
            lg = fv.feed(feed, n_score)
            self._v_len[rpx] = fv.valid_len
        else:
            from transformers import DynamicCache
            cache = self._v_cache.get(rpx)
            if cache is None:
                cache = DynamicCache()
                self._v_cache[rpx] = cache
                self._v_len[rpx] = 0
            vlen = self._v_len[rpx]
            feed = realized[vlen:] + drafts[:-1]
            ids = torch.tensor([feed], dtype=torch.long, device=device)
            pos = torch.arange(vlen, vlen + len(feed), device=device).unsqueeze(0)
            out = self._verifier(input_ids=ids, position_ids=pos,
                                 past_key_values=cache, use_cache=True)
            lg = out.logits[0, -n_score:].float()
            self._v_len[rpx] = vlen + len(feed)
        if self.verifier_temp != 1.0:
            lg = lg / self.verifier_temp
        lp = torch.log_softmax(lg, dim=-1)
        tgt = torch.tensor(drafts[1:], dtype=torch.long, device=lg.device)
        p_tilde = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).exp()
        self._stats["verifier_calls"] += 1
        self._stats["verifier_time_ms"] += (time.perf_counter() - t0) * 1e3
        return p_tilde

    def _crop_verifier(self, rpx, keep_len: int):
        if self._fv is not None and hasattr(self._fv, "crop"):
            # multi-request verifier: per-rpx crop, no owner dance
            if self._v_len.get(rpx, 0) > keep_len:
                self._fv.crop(rpx, keep_len)
                self._v_len[rpx] = keep_len
            return
        if self._fv is not None and self._fv.owner == rpx:
            if self._v_len.get(rpx, 0) > keep_len:
                self._fv.rewind(keep_len)
                self._v_len[rpx] = keep_len
            return
        cache = self._v_cache.get(rpx)
        if cache is not None and self._v_len.get(rpx, 0) > keep_len:
            cache.crop(keep_len)
            self._v_len[rpx] = keep_len

    def _rpx(self, forward_batch, bid: int):
        try:
            return int(forward_batch.req_pool_indices[bid].item())
        except Exception:
            return None

    def _accept_ok(self, depth: int, p_t: float, q: float) -> bool:
        """depth = 2.. for the first judged proposed token (y1 is free)."""
        if self.deep_start and depth >= self.deep_start:
            return p_t >= self.theta_deep          # deep slots: verifier-led
        if self.rule == "ptilde":
            return p_t >= self.theta
        return p_t * q >= self.theta               # "and"

    # ---------------- main: ONE forward per cycle ----------------

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        blk = self.block_size

        # Create the verifier BEFORE the first target forward ever runs: its CUDA
        # graph capture must not interleave with target graph REPLAYS (capturing in a
        # process with actively-replaying graphs corrupts memory — bisected 2026-07-31).
        # At the first run() call the target's graphs exist but have never replayed.
        if self._fv is None and self._verifier is None:
            self._load_verifier(device)

        self._dllm_write_override = getattr(self, "_dllm_write_override", {})
        self._advance_override = getattr(self, "_advance_override", {})
        self._kv_trim_info = getattr(self, "_kv_trim_info", {})
        self._dllm_write_override.clear()
        self._advance_override.clear()
        self._kv_trim_info.clear()

        mask_index = forward_batch.input_ids == self.mask_id
        if os.environ.get("MARSV_CYCLE_LOG"):
            self._cyc_t0 = time.perf_counter()
            _prev = getattr(self, "_cyc_t3_prev", None)
            self._cyc_gap = (self._cyc_t0 - _prev) if _prev else None
        if os.environ.get("MARSV_LOG"):
            self._entry_logs = getattr(self, "_entry_logs", 0)
            if self._entry_logs < 8:
                self._entry_logs += 1
                logger.info(f"[marsvS entry] mode={forward_batch.forward_mode} "
                            f"n_ids={forward_batch.input_ids.numel()} bs={batch_size} "
                            f"n_mask={int(torch.sum(mask_index).item())} "
                            f"ext={forward_batch.extend_seq_lens}")
        # ---- prefill fast path: capture prompt ids + the t0 seed ----
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
            if os.environ.get("MARSV_LOG"):
                logger.info(f"[marsvS prefill] fl={'None' if fl is None else tuple(fl.shape)} "
                            f"lens={lens} mode={forward_batch.forward_mode} "
                            f"rpx={[self._rpx(forward_batch, b) for b in range(len(lens or []))]}")
            if fl is not None and lens is not None:
                off = 0
                for bid, ln in enumerate(lens):
                    rpx = self._rpx(forward_batch, bid)
                    if rpx is not None and ln > 0:
                        self._pending[rpx] = int(torch.argmax(fl[off + ln - 1]).item())
                    off += ln
            self._cyc_t3_prev = None       # request boundary: the next cycle gap must not include prefill / client time
            return out.logits_output, [], out.can_run_graph

        # ---- classify: write re-feed tokens into the all-mask window ----
        # Window templates that can arrive here:
        #   [k refeed slots as masks][...masks]   normal round: write rf into the front
        #   [start REAL prompt toks][...masks]    FINAL PREFILL round: the last P%16
        #       prompt tokens ride in the window (short prompts: the whole prompt).
        #       They are context, not emissions: k=start, nothing written, no audit.
        _el = forward_batch.extend_seq_lens
        ext = (_el.tolist() if isinstance(_el, torch.Tensor) else list(_el)) if _el is not None else [blk] * batch_size
        base = [0] * batch_size
        for _b in range(1, batch_size):
            base[_b] = base[_b - 1] + ext[_b - 1]
        refeed_k = [0] * batch_size
        prompt_round = [False] * batch_size
        inline_seg = {}            # bid -> segment length of an inline-prefill (mask-free) segment
        for bid in range(batch_size):
            bs_ = base[bid]
            seg = ext[bid] if bid < len(ext) else blk   # inline-prefill segments carry the whole (remaining) prompt, not one block
            block_ids = forward_batch.input_ids[bs_:bs_ + seg]
            rpx = self._rpx(forward_batch, bid)
            if rpx is None:
                continue
            rf = self._refeed.get(rpx)
            if rf is None:
                bm = block_ids == self.mask_id
                n_mask_blk = int(torch.sum(bm).item())
                start = blk - n_mask_blk
                if n_mask_blk == 0:
                    inline_seg[bid] = seg
                    # FULL window of committed prompt: a chunked-prefill segment
                    # riding in a mixed batch (at C>1 the whole-batch mask-free fast
                    # path cannot trigger). Advance-only round: capture prefix; the
                    # seed comes from rows[blk-1] in pass 1; emit nothing.
                    _full = block_ids.tolist()
                    _pf = self._prefix_ids.setdefault(rpx, [])
                    if _pf[-len(_full):] != _full:
                        _pf.extend(_full)
                    self._pending.pop(rpx, None)
                    self._emit_refeed[rpx] = False
                    prompt_round[bid] = True
                    refeed_k[bid] = blk
                    continue
                if start > 0 and bool(bm[start:].all().item()):
                    # final-prefill mixed window: real prefix is committed context.
                    # DEDUP: when the whole-batch prefill fast path already captured
                    # the full prompt (C=1), these start tokens are its tail — do not
                    # append them twice to the verifier context.
                    rf = block_ids[:start].tolist()
                    _pf = self._prefix_ids.setdefault(rpx, [])
                    if _pf[-start:] != rf:
                        _pf.extend(rf)
                    self._pending.pop(rpx, None)
                    self._emit_refeed[rpx] = False
                    self._refeed[rpx] = rf
                    prompt_round[bid] = True
                    refeed_k[bid] = start
                    continue                        # tokens already real; nothing to write
                pend = self._pending.pop(rpx, None)
                if pend is None:
                    # no seed (shouldn't happen after prefill) — degrade to 1 mask AR
                    if os.environ.get("MARSV_LOG"):
                        logger.warning(f"[marsvS classify] NO SEED rpx={rpx} "
                                       f"pending_keys={list(self._pending.keys())} "
                                       f"prefix_len={len(self._prefix_ids.get(rpx, []))}")
                    refeed_k[bid] = 0
                    continue
                rf = [pend]
                self._refeed[rpx] = rf
                self._emit_refeed[rpx] = True
                # the t0 seed is realized context for the verifier too
                self._prefix_ids.setdefault(rpx, []).append(pend)
            k = min(len(rf), blk - 1)              # always leave >=1 mask
            for i in range(k):
                block_ids[i] = rf[i]
            refeed_k[bid] = k

        # ---- ONE forward ----
        forward_batch.dllm_force_causal = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        forward_batch.dllm_force_causal = False
        if overlap_fn is not None:
            overlap_fn()
        logits_output = out.logits_output
        full_logits = logits_output.full_logits
        self._stats["total_forwards"] += 1

        req_to_token = model_runner.req_to_token_pool.req_to_token
        _sl = forward_batch.seq_lens
        seq_lens_cpu = _sl.tolist() if isinstance(_sl, torch.Tensor) else list(_sl)

        next_token_ids_list = []
        mars_ws_list = []
        all_trim_rpx, all_trim_pos, trim_counts = [], [], {}

        # ---- PASS 1: per-request proposed-block extraction (no verifier calls) ----
        staged = {}                                # bid -> dict for pass 2
        score_items = []                           # batched verifier inputs
        score_bids = []
        for bid in range(batch_size):
            bs_ = base[bid]
            rpx = self._rpx(forward_batch, bid)
            k = refeed_k[bid]
            rows = full_logits[bs_:bs_ + blk]      # row r predicts position r+1
            rf = self._refeed.get(rpx, []) if rpx is not None else []

            # ---- free audit: re-fed rows give exact AR for c2..ck ----
            # (skip prompt rounds: the real prefix is prompt text, not accepted proposed tokens)
            if self.audit and k >= 2 and not prompt_round[bid]:
                ar_pred = rows[:k - 1].argmax(dim=-1).tolist()
                for i in range(k - 1):
                    self._stats["audit_pairs"] += 1
                    if ar_pred[i] != rf[i + 1]:
                        self._stats["audit_mismatch"] += 1

            # ---- proposed block: y1 (exact, from ck's row) + mask-row chain ----
            n_drafts = blk - k                     # y1..y_{blk-k}
            draft_rows = rows[k - 1:blk - 1] if k >= 1 else rows[:blk - 1]
            if k == 0:
                staged[bid] = None                 # degenerate no-seed window
                continue
            if bid in inline_seg or k == blk:
                # advance-only prompt round: seed the next window from the last
                # real row of the segment (exact AR), emit nothing, keep all KV
                if rpx is not None:
                    self._pending[rpx] = int(torch.argmax(full_logits[bs_ + inline_seg.get(bid, blk) - 1]).item())
                    self._refeed.pop(rpx, None)
                staged[bid] = {"advance_only": True, "rpx": rpx, "k": blk}
                continue
            if self.seq_head_path and self._sh_W1 is None:
                import torch as _t
                _ck = _t.load(self.seq_head_path, map_location=device)
                self._sh_W1 = _ck["W1"].to(device=device, dtype=torch.float32)
                self._sh_W2 = _ck["W2"].to(device=device, dtype=torch.float32)
                logger.info(f"[MarsVerifierStream] seq head loaded {self.seq_head_path}")
            if self._sh_W1 is not None and n_drafts > 1:
                # y1 (row k-1) is a REAL row -> exact AR, no bias. Mask rows get the
                # sequential bigram bias conditioned on the PREVIOUS proposed token, with the
                # exact-Z top-K correction (K=64): tail mass stays untouched.
                fr = draft_rows.float()
                topv, topi = fr.topk(64, dim=-1)
                lse_full = fr.logsumexp(dim=-1)
                tail = (lse_full.exp() - topv.exp().sum(-1)).clamp_min(0)
                # sequential chain kept ON-DEVICE: no host syncs inside the loop
                d_list = [topi[0, 0]]
                q_list = [(topv[0, 0] - lse_full[0]).exp()]
                prev_t = topi[0, 0]
                for j in range(1, n_drafts):
                    b = self._sh_W1[prev_t] @ self._sh_W2[:, topi[j]]   # [64]
                    corr = topv[j] + b
                    jj = corr.argmax()
                    d_list.append(topi[j, jj])
                    # SELECTION by corrected logits, GATE q stays the RAW proposal
                    # prob of the selected token — the head must not loosen the gate
                    # (corrected q is sharper and de-calibrates theta; measured).
                    e_raw = topv[j].exp()
                    q_list.append(e_raw[jj] / (e_raw.sum() + tail[j]))
                    prev_t = topi[j, jj]
                drafts = torch.stack(d_list).tolist()      # ONE sync
                qs = torch.stack(q_list).tolist()
                dprobs = None                      # no full proposer row on this path
            else:
                probs = F.softmax(draft_rows.float(), dim=-1)
                xs = probs.argmax(dim=-1)
                qs = probs.gather(1, xs.unsqueeze(1)).squeeze(1).tolist()
                drafts = xs.tolist()               # y1..y_{n_drafts}
                dprobs = probs                     # kept for the vtok="pq" rescoring
            # hybrid pre-walk: count leading proposed tokens that pass on q alone
            free_run = 0
            if self.q_hi < 1.0:
                for i in range(n_drafts - 1):
                    if qs[i + 1] >= self.q_hi:
                        free_run += 1
                    else:
                        break
            staged[bid] = {"rpx": rpx, "k": k, "rf": rf, "qs": qs,
                           "drafts": drafts, "n_drafts": n_drafts,
                           "free_run": free_run, "p_tilde": None,
                           "draft_probs": (dprobs if (self.vtok == "pq" or self.joint) else None)}
            need_verify = n_drafts > 1 and free_run < n_drafts - 1
            if need_verify and rpx is not None:
                realized = self._prefix_ids.get(rpx, [])
                vlen = self._v_len.get(rpx, 0)
                score_items.append({
                    "rpx": rpx, "feed": realized[vlen:] + drafts[:-1],
                    "need": n_drafts - 1, "targets": drafts[1:],
                    "temp": self.verifier_temp, "_realized_len": len(realized)})
                score_bids.append(bid)

        # ---- batched verify: ONE verifier replay for every request's proposed block ----
        if os.environ.get("MARSV_CYCLE_LOG"):
            torch.cuda.synchronize()
            self._cyc_t1 = time.perf_counter()
        if score_items:
            t0 = time.perf_counter()
            self._load_verifier(device)
            fv = self._fv
            if fv is not None and hasattr(fv, "batched_score"):
                p_list = fv.batched_score(score_items)
                vtoks = getattr(fv, "_last_tokens", {})
                vtopk = getattr(fv, "_last_topk", {})
                for it, bid, pt in zip(score_items, score_bids, p_list):
                    staged[bid]["p_tilde"] = pt
                    staged[bid]["vtok_ids"] = vtoks.get(it["rpx"])
                    staged[bid]["vtok_topk"] = vtopk.get(it["rpx"])
                    if pt is not None:
                        self._v_len[it["rpx"]] = fv.vlen(it["rpx"])
            else:
                for it, bid in zip(score_items, score_bids):
                    staged[bid]["p_tilde"] = self._score_chain(
                        it["rpx"], self._prefix_ids.get(it["rpx"], []),
                        staged[bid]["drafts"], device)
            self._stats["verifier_calls"] += 1
            self._stats["verifier_time_ms"] += (time.perf_counter() - t0) * 1e3
        if os.environ.get("MARSV_CYCLE_LOG"):
            torch.cuda.synchronize()
            self._cyc_t2 = time.perf_counter()

        # ---- PASS 2: accept/emit/state, in bid order ----
        for bid in range(batch_size):
            rpx = self._rpx(forward_batch, bid)
            st = staged.get(bid)
            if st is None:
                # degenerate no-seed window: cannot predict slot 0 -> emit nothing
                next_token_ids_list.append([])
                mars_ws_list.append([])
                trim_counts[bid] = blk
                if rpx is not None:
                    sl = seq_lens_cpu[bid]
                    for t in range(blk):
                        all_trim_rpx.append(rpx)
                        all_trim_pos.append(sl - 1 - t)
                self._advance_override[rpx] = 0
                continue
            if st.get("advance_only"):
                next_token_ids_list.append([])
                mars_ws_list.append([])
                trim_counts[bid] = 0
                if rpx is not None:
                    self._advance_override[rpx] = inline_seg.get(bid, blk)
                continue
            k, rf = st["k"], st["rf"]
            qs, drafts, n_drafts = st["qs"], st["drafts"], st["n_drafts"]
            p_tilde = st["p_tilde"]

            accept_n = 1
            joint_corr = None
            free_run = st.get("free_run", 0)
            if p_tilde is not None:
                # ONE device sync for the whole chain. This used to be float(p_tilde[i]) inside
                # the loop, i.e. one D2H sync per accepted token, and each of those drains the
                # queue on the critical path between the verifier and the next proposing pass.
                pt = p_tilde.tolist() if hasattr(p_tilde, "tolist") else list(p_tilde)
                tk_j = st.get("vtok_topk") if self.joint else None
                pr_j = st.get("draft_probs") if self.joint else None
                if tk_j is not None and pr_j is not None and n_drafts > 1:
                    # joint decode (see __init__): verifier row i judges drafts[i+1], whose
                    # proposer distribution is row i+1 of draft_probs. One device sync.
                    ids_all, qv_all = tk_j[0], tk_j[1]                     # [R, K]
                    R = min(int(ids_all.shape[0]), n_drafts - 1)
                    V = int(pr_j.shape[1])
                    if os.environ.get("MARSV_JOINT_DEBUG"):
                        torch.cuda.synchronize()
                        logger.info(f"[joint-pre] rpx={rpx} R={R} ids={tuple(ids_all.shape)}/{ids_all.dtype}/{ids_all.device} "
                                    f"qv={tuple(qv_all.shape)}/{qv_all.dtype}/{qv_all.device} pr={tuple(pr_j.shape)}/{pr_j.dtype}/{pr_j.device} "
                                    f"n_drafts={n_drafts} accept_n={accept_n}")
                    ids_r = ids_all[:R].to(device=pr_j.device, dtype=torch.long).clamp_(0, V - 1)
                    pv_all = pr_j[1:R + 1].gather(1, ids_r)
                    jt = pv_all * qv_all[:R].to(device=pr_j.device, dtype=pv_all.dtype)
                    jmax, jarg = jt.max(dim=1)
                    jtok = ids_r.gather(1, jarg.unsqueeze(1)).squeeze(1)
                    jmax_l, jtok_l = jmax.tolist(), jtok.tolist()
                    if os.environ.get("MARSV_JOINT_DEBUG"):
                        torch.cuda.synchronize()
                        logger.info(f"[joint] rpx={rpx} R={R} K={int(ids_all.shape[1])} V={V} pr={tuple(pr_j.shape)} "
                                    f"ids[min,max]=({int(ids_all[:R].min())},{int(ids_all[:R].max())}) "
                                    f"drafts={drafts[:R+1]} jtok={jtok_l} jmax={[round(x,3) for x in jmax_l]}")
                    for i in range(R):
                        if jmax_l[i] < self.theta:
                            self._stats["v_reject"] += 1
                            break
                        if jtok_l[i] == drafts[i + 1]:
                            accept_n += 1
                            self._stats["v_accept"] += 1
                        else:
                            joint_corr = int(jtok_l[i])
                            self._stats.setdefault("joint_corr", 0)
                            self._stats["joint_corr"] += 1
                            break
                else:
                    for i in range(n_drafts - 1):
                        if (i < free_run) or self._accept_ok(
                                i + 2, pt[i], qs[i + 1]):
                            accept_n += 1
                            self._stats["v_accept"] += 1
                        else:
                            self._stats["v_reject"] += 1
                            break
                realized_len = len(self._prefix_ids.get(rpx, []))
                self._crop_verifier(
                    rpx, realized_len + min(accept_n, n_drafts - 1))
            elif free_run > 0:
                # verifier skipped: the whole q-confident run auto-commits
                accept_n = 1 + free_run
                self._stats["v_accept"] += free_run
                self._stats.setdefault("gate_skips", 0)
                self._stats["gate_skips"] += 1

            accepted = drafts[:accept_n]
            if joint_corr is not None and len(rf) + accept_n + 1 <= blk:
                # joint mode: the product preferred another candidate at the stop position
                accepted = accepted + [joint_corr]
                self._stats.setdefault("joint_emitted", 0)
                self._stats["joint_emitted"] += 1
            # vtok: the gate stopped before drafts[accept_n], and row accept_n-1 of the
            # verifier's scores is its own distribution over that same position given the
            # accepted prefix. Committing its argmax buys one token per rejected cycle.
            if (self.vtok and p_tilde is not None and accept_n <= n_drafts - 1
                    and len(rf) + accept_n + 1 <= blk):
                tok = None
                if self.vtok == "pq":
                    tk = st.get("vtok_topk")
                    pr = st.get("draft_probs")
                    if tk is not None and pr is not None and accept_n - 1 < int(tk[0].shape[0]):
                        ids, qv = tk[0][accept_n - 1], tk[1][accept_n - 1]
                        # p_tilde of those same candidates, from the proposer's own row at
                        # this slot; the gate's score is the product, so commit its argmax
                        pv = pr[accept_n].index_select(0, ids)
                        tok = int(ids[int(torch.argmax(pv * qv))])
                if tok is None:
                    vt = st.get("vtok_ids")
                    if vt is not None and accept_n - 1 < int(vt.shape[0]):
                        tok = int(vt[accept_n - 1])
                if tok is not None:
                    accepted = accepted + [tok]
                    self._stats.setdefault("vtok_emitted", 0)
                    self._stats["vtok_emitted"] += 1
            emit = (list(rf) if self._emit_refeed.get(rpx, False) else []) + accepted
            ws = ([1] * len(rf) if self._emit_refeed.get(rpx, False) else []) \
                + [1] + [0] * (len(accepted) - 1)   # y1 clean-conditioned
            next_token_ids_list.append(emit)
            mars_ws_list.append(ws)

            # ---- state + overrides ----
            if rpx is not None:
                self._emit_refeed[rpx] = False
                self._refeed[rpx] = accepted
                self._prefix_ids.setdefault(rpx, []).extend(accepted)
                dllm_tokens = list(rf) + accepted
                dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))
                self._dllm_write_override[rpx] = dllm_tokens
                self._advance_override[rpx] = k
                tc = blk - k                        # every mask row's KV is garbage
                trim_counts[bid] = tc
                sl = seq_lens_cpu[bid]
                for t in range(tc):
                    all_trim_rpx.append(rpx)
                    all_trim_pos.append(sl - 1 - t)

            self._stats["total_tokens"] += len(emit)

        if os.environ.get("MARSV_CYCLE_LOG"):
            torch.cuda.synchronize()
            _t3 = time.perf_counter()
            self._cyc_t3_prev = _t3
            # aggregate timeline (same layout as [marsX timeline]): gap = scheduler time between cycles
            _acc = getattr(self, "_cyc_acc", None)
            if _acc is None:
                _acc = {"n": 0, "gap": 0.0, "fwdp1": 0.0, "verify": 0.0, "p2": 0.0}; self._cyc_acc = _acc
            _t0_ = getattr(self, "_cyc_t0", None)
            if _t0_ is not None and getattr(self, "_cyc_gap", None) is not None:
                _acc["n"] += 1; _acc["gap"] += self._cyc_gap; _acc["fwdp1"] += (self._cyc_t1 - _t0_)
                _acc["verify"] += (self._cyc_t2 - self._cyc_t1); _acc["p2"] += (_t3 - self._cyc_t2)
                if _acc["n"] % 25 == 0:
                    n_ = _acc["n"]
                    logger.info("[marsvS timeline] n=%d  gap(sched)=%.2fms fwd+p1=%.2fms verify=%.2fms p2=%.2fms  cycle=%.2fms" % (
                        n_, 1e3*_acc["gap"]/n_, 1e3*_acc["fwdp1"]/n_, 1e3*_acc["verify"]/n_, 1e3*_acc["p2"]/n_,
                        1e3*(_acc["gap"]+_acc["fwdp1"]+_acc["verify"]+_acc["p2"])/n_))
            self._cyc_n = getattr(self, "_cyc_n", 0) + 1
            if self._cyc_n <= int(os.environ.get("MARSV_CYCLE_LOG_MAX", "80")):   # diagnostic only; unset MARSV_CYCLE_LOG = no effect
                _t0 = getattr(self, "_cyc_t0", None)
                logger.info(
                    f"[marsvS cycle {self._cyc_n}] bs={batch_size} "
                    f"fwd+p1={(self._cyc_t1 - _t0)*1e3 if _t0 else -1:.1f}ms "
                    f"verify={(self._cyc_t2 - self._cyc_t1)*1e3:.1f}ms "
                    f"p2={(_t3 - self._cyc_t2)*1e3:.1f}ms "
                    f"emits={[len(x) for x in next_token_ids_list]} "
                    f"scored={len(score_items)} "
                    f"vavail={getattr(getattr(self._fv, 'mr', None), 'token_to_kv_pool_allocator', None).available_size() if hasattr(self._fv, 'mr') else -1} "
                    f"rpxs={[self._rpx(forward_batch, b) for b in range(batch_size)]} "
                    f"ks={[refeed_k[b] for b in range(batch_size)]} "
                    f"npi={forward_batch.req_pool_indices.shape if hasattr(forward_batch.req_pool_indices, 'shape') else len(forward_batch.req_pool_indices)}")

        # ---- batched KV-trim index lookup ----
        if all_trim_rpx:
            all_kv = req_to_token[all_trim_rpx, all_trim_pos]
        else:
            all_kv = None
        off = 0
        for bid in range(batch_size):
            rpx = self._rpx(forward_batch, bid)
            if rpx is None:
                continue
            tc = trim_counts.get(bid, 0)
            self._kv_trim_info[rpx] = {
                "kv_indices_gpu": all_kv[off:off + tc] if all_kv is not None else None,
                "trim_count": tc,
            }
            off += tc

        logits_output.customized_info = {"mars_ws": mars_ws_list}
        return logits_output, next_token_ids_list, out.can_run_graph


Algorithm = MarsVerifierStream
