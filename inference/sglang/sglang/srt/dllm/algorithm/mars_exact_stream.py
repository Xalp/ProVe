"""MarsExactStream — ProVe-lossless: self-verification, TWO proposer forwards per cycle, no verifier model.

Forward A (propose): window at S..S+blk-1 = [k re-fed real tokens][blk-k masks]; the right-shifted
                    readout gives y_1 (row k-1, exact AR) and y_2..y_{blk-k} (mask rows).
Forward B (verify): the same window with the masks OVERWRITTEN by y_1..y_{blk-k} (all real tokens,
                    causal). Row k-1+i now predicts position k+i+1 under EXACT conditioning, i.e.
                    g_{i+1} = argmax of p(. | context, y_1..y_i); row k-1 gives g_1 (= y_1).
Accept:             j = longest leading run with y_i == g_i; correction t = g_{j+1}.
Emit:               y_1..y_j, t  (never empty: at least the exact token t).
KV:                 slots of y_1..y_j were computed from REAL tokens in forward B -> clean, kept;
                    the blk-k-j slots after them are trimmed; t has no KV yet and is re-fed as
                    the single real token at the head of the next window (advance = k + j).
The proposer verifies its own proposed block: greedy output equals greedy AR decoding of the
proposer token for token (up to bf16 tie-breaking). Cost per cycle is two full forwards, so
tokens/forward = (j+1)/2, at most blk/2.

Config keys: block_size (window; proposes blk-1 tokens per cycle), audit (bool: count g_1 != y_1).
The variables named `drafts` below hold the proposed block.
Greedy only. Diagnostics: MARSV_LOG (cum stats at request cleanup + entry logs).
"""
import logging
import os
import time
from typing import List, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class MarsExactStream(DllmAlgorithm):

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.audit = bool(config.algorithm_config.get("audit", True))
        self.theta = float(config.algorithm_config.get("theta", 0.0))   # 0 = exact (unchanged)
        # Which lossy gate theta scores against. "q" = the verifier's own probability for the
        # proposed token. "pq" = p_tilde * q, the same product the small-verifier configuration
        # deploys, so one rule spans the whole verifier-size dial. Default "q" keeps every run
        # submitted before 2026-09-20 reproducible.
        self.gate = str(config.algorithm_config.get("gate", "q")).lower()
        self._refeed = {}          # rpx -> [t] the correction token to re-feed (or prompt tail)
        self._emit_refeed = {}     # rpx -> True only for the first-round t0 seed
        self._pending = {}         # rpx -> t0 from prefill logits
        self._prefix_len = {}      # rpx -> committed length (bookkeeping only)
        for k in ("cycles", "verify_forwards", "draft_match", "draft_reject",
                  "full_runs", "seed_mismatch", "total_forwards", "total_tokens",
                  "tA_ms", "nA", "tB_ms", "nB"):
            self._stats.setdefault(k, 0)
        self._pass_log = bool(os.environ.get("MARSV_PASS_LOG"))   # per-pass GPU time (diagnostic; off = no effect)

    def cleanup_request(self, req_pool_idx: int):
        for d in (self._refeed, self._emit_refeed, self._pending, self._prefix_len):
            d.pop(req_pool_idx, None)
        if os.environ.get("MARSV_LOG"):
            s = self._stats
            logger.info(
                f"[MarsExactStream] cum stats: fwd={s['total_forwards']} "
                f"tok={s['total_tokens']} "
                f"tok/fwd={s['total_tokens']/max(1,s['total_forwards']):.3f} "
                f"cycles={s['cycles']} tok/cycle={s['total_tokens']/max(1,s['cycles']):.3f} "
                f"match={s['draft_match']} reject={s['draft_reject']} "
                f"full_runs={s['full_runs']} seed_mismatch={s['seed_mismatch']}"
                + (f" passA={s['tA_ms']/max(1,s['nA']):.2f}ms/{s['nA']} passB={s['tB_ms']/max(1,s['nB']):.2f}ms/{s['nB']}" if self._pass_log else ""))

    def _rpx(self, forward_batch, bid: int):
        try:
            return int(forward_batch.req_pool_indices[bid].item())
        except Exception:
            return None

    # ---- MARSV_CYCLE_TIMELINE=1: per-cycle wall-clock breakdown (diagnostic; syncs the GPU at each mark) ----
    def _tl_mark(self, name):
        if not getattr(self, "_tl_on", False):
            return
        torch.cuda.synchronize()
        self._tl_marks[name] = time.perf_counter()

    def _tl_finish(self):
        m = self._tl_marks
        if "exit" not in m or "A0" not in m or "B1" not in m:
            self._tl_last_exit = None      # incomplete cycle (prefill / advance-only): do not count the next gap
            return
        acc = self._tl_acc
        acc["n"] += 1
        acc["gap"] += (m["entry"] - self._tl_last_exit) if self._tl_last_exit else 0.0
        acc["prep"] += m["A0"] - m["entry"]; acc["fwdA"] += m["A1"] - m["A0"]
        acc["mid"] += m["B0"] - m["A1"]; acc["fwdB"] += m["B1"] - m["B0"]; acc["post"] += m["exit"] - m["B1"]
        self._tl_last_exit = m["exit"]
        if acc["n"] % 25 == 0:
            n = acc["n"]
            logger.info("[marsX timeline] n=%d  gap(sched)=%.2fms prep=%.2fms fwdA=%.2fms mid=%.2fms fwdB=%.2fms post=%.2fms  cycle=%.2fms" % (
                n, 1e3*acc["gap"]/n, 1e3*acc["prep"]/n, 1e3*acc["fwdA"]/n, 1e3*acc["mid"]/n, 1e3*acc["fwdB"]/n, 1e3*acc["post"]/n,
                1e3*(acc["gap"]+acc["prep"]+acc["fwdA"]+acc["mid"]+acc["fwdB"]+acc["post"])/n))

    # ---------------- main: TWO forwards per cycle ----------------

    @torch.no_grad()
    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        blk = self.block_size
        if not hasattr(self, "_tl_on"):
            self._tl_on = bool(os.environ.get("MARSV_CYCLE_TIMELINE")); self._tl_marks = {}; self._tl_last_exit = None
            self._tl_acc = {k: 0.0 for k in ("gap", "prep", "fwdA", "mid", "fwdB", "post")}; self._tl_acc["n"] = 0
        self._tl_marks = {}; self._tl_mark("entry")

        self._dllm_write_override = getattr(self, "_dllm_write_override", {})
        self._advance_override = getattr(self, "_advance_override", {})
        self._kv_trim_info = getattr(self, "_kv_trim_info", {})
        self._dllm_write_override.clear()
        self._advance_override.clear()
        self._kv_trim_info.clear()

        mask_index = forward_batch.input_ids == self.mask_id
        if os.environ.get("MARSV_LOG"):
            self._entry_logs = getattr(self, "_entry_logs", 0)
            if self._entry_logs < 8:
                self._entry_logs += 1
                logger.info(f"[marsX entry] mode={forward_batch.forward_mode} "
                            f"n_ids={forward_batch.input_ids.numel()} bs={batch_size} "
                            f"n_mask={int(torch.sum(mask_index).item())} "
                            f"ext={forward_batch.extend_seq_lens}")

        # ---- prefill fast path (whole batch mask-free): one forward, seed t0 ----
        if torch.sum(mask_index).item() == 0:
            _el = forward_batch.extend_seq_lens
            lens = None
            if _el is not None:
                lens = _el.tolist() if isinstance(_el, torch.Tensor) else list(_el)
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            self._stats["total_forwards"] += 1
            fl = getattr(out.logits_output, "full_logits", None)
            if fl is not None and lens is not None:
                off = 0
                for bid, ln in enumerate(lens):
                    rpx = self._rpx(forward_batch, bid)
                    if rpx is not None and ln > 0:
                        self._pending[rpx] = int(torch.argmax(fl[off + ln - 1]).item())
                        self._prefix_len[rpx] = self._prefix_len.get(rpx, 0) + ln
                    off += ln
            if getattr(self, "_tl_on", False): self._tl_last_exit = None
            return out.logits_output, [], out.can_run_graph

        # ---- per-request window offsets (uniform blk stride in pure decode) ----
        _el = forward_batch.extend_seq_lens
        if _el is not None:
            ext = _el.tolist() if isinstance(_el, torch.Tensor) else list(_el)
        else:
            ext = [blk] * batch_size
        base = [0] * batch_size
        for bid in range(1, batch_size):
            base[bid] = base[bid - 1] + ext[bid - 1]

        # ---- classify: write re-feed tokens into the all-mask window ----
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
                    # mask-free segment: an inline-prefill prompt (any length) or a full window of committed
                    # prompt; advance-only round, seed from the LAST row of the segment, emit nothing
                    inline_seg[bid] = seg
                    self._pending.pop(rpx, None)
                    self._emit_refeed[rpx] = False
                    prompt_round[bid] = True
                    refeed_k[bid] = blk
                    continue
                if start > 0 and bool(bm[start:].all().item()):
                    # final-prefill mixed window: the real prefix is committed context
                    rf = block_ids[:start].tolist()
                    self._pending.pop(rpx, None)
                    self._emit_refeed[rpx] = False
                    self._refeed[rpx] = rf
                    prompt_round[bid] = True
                    refeed_k[bid] = start
                    continue
                pend = self._pending.pop(rpx, None)
                if pend is None:
                    if os.environ.get("MARSV_LOG"):
                        logger.warning(f"[marsX classify] NO SEED rpx={rpx}")
                    refeed_k[bid] = 0
                    continue
                rf = [pend]
                self._refeed[rpx] = rf
                self._emit_refeed[rpx] = True
            k = min(len(rf), blk - 1)              # always leave >= 1 mask
            for i in range(k):
                block_ids[i] = rf[i]
            refeed_k[bid] = k

        # ---- forward A: propose ----
        forward_batch.dllm_force_causal = True
        if self._pass_log:
            torch.cuda.synchronize(); _tA = time.perf_counter()
        self._tl_mark("A0")
        outA = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        self._tl_mark("A1")
        if self._pass_log:
            torch.cuda.synchronize(); self._stats["tA_ms"] += (time.perf_counter() - _tA) * 1000.0; self._stats["nA"] += 1
        self._stats["total_forwards"] += 1
        fullA = outA.logits_output.full_logits

        staged = {}
        need_b = False
        for bid in range(batch_size):
            bs_ = base[bid]
            rpx = self._rpx(forward_batch, bid)
            k = refeed_k[bid]
            if rpx is None or k == 0:
                staged[bid] = None
                continue
            rows = fullA[bs_:bs_ + blk]            # row r predicts position r+1
            if bid in inline_seg or k == blk:
                self._pending[rpx] = int(torch.argmax(fullA[bs_ + inline_seg.get(bid, blk) - 1]).item())
                self._refeed.pop(rpx, None)
                staged[bid] = {"advance_only": True}
                continue
            n_drafts = blk - k                     # y_1..y_{n_drafts}
            drafts_gpu = rows[k - 1:blk - 1].argmax(dim=-1)    # y_1 from the last REAL row
            # p_tilde: the PROPOSER's own probability for the token it just proposed. Because the
            # draft IS the argmax of these rows, this is the proposer's top-1 confidence, i.e. the
            # same p_tilde the small-verifier configuration multiplies into its gate. Computed here
            # while rows/drafts are both live; only when the lossy p_tilde*q gate is selected.
            ptil = None
            if self.theta > 0.0 and self.gate == "pq":
                ptil = rows[k - 1:blk - 1].float().softmax(dim=-1).gather(
                    1, drafts_gpu.unsqueeze(1)).squeeze(1)
            # write the proposed tokens over the mask slots for forward B
            forward_batch.input_ids[bs_ + k:bs_ + blk] = drafts_gpu
            staged[bid] = {"k": k, "n_drafts": n_drafts, "drafts_gpu": drafts_gpu, "ptil": ptil}
            need_b = True

        # ---- forward B: verify the proposed tokens under exact conditioning ----
        if need_b:
            if self._pass_log:
                torch.cuda.synchronize(); _tB = time.perf_counter()
            self._tl_mark("B0")
            outB = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            self._tl_mark("B1")
            if self._pass_log:
                torch.cuda.synchronize(); self._stats["tB_ms"] += (time.perf_counter() - _tB) * 1000.0; self._stats["nB"] += 1
            self._stats["total_forwards"] += 1
            self._stats["verify_forwards"] += 1
            out = outB
        else:
            out = outA
        forward_batch.dllm_force_causal = False
        if overlap_fn is not None:
            overlap_fn()
        logits_output = out.logits_output
        fullB = logits_output.full_logits

        req_to_token = model_runner.req_to_token_pool.req_to_token
        _sl = forward_batch.seq_lens
        seq_lens_cpu = _sl.tolist() if isinstance(_sl, torch.Tensor) else list(_sl)

        next_token_ids_list = []
        mars_ws_list = []
        all_trim_rpx, all_trim_pos, trim_counts = [], [], {}

        # ---- accept / emit / state ----
        for bid in range(batch_size):
            bs_ = base[bid]
            rpx = self._rpx(forward_batch, bid)
            st = staged.get(bid)
            if st is None:
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
                self._advance_override[rpx] = inline_seg.get(bid, blk)
                continue
            k, n_drafts = st["k"], st["n_drafts"]
            rf = self._refeed.get(rpx, [])
            rowsB = fullB[bs_:bs_ + blk]
            g_gpu = rowsB[k - 1:blk].argmax(dim=-1)            # g_1..g_{n_drafts+1}
            y_gpu = st["drafts_gpu"]                            # y_1..y_{n_drafts}
            qy = None
            if self.theta > 0.0:
                # LOSSY dial: keep a proposed token while the verifier's own probability for it
                # clears theta, instead of demanding it be the argmax. The verification pass is
                # teacher-forced on the PROPOSED tokens, so row i is conditioned on y_1..y_i and
                # the relaxed accept stays correctly conditioned. The correction token is still
                # the verifier's argmax, i.e. still its own token.
                # 2026-09-20: y_gpu used to be READ here and ASSIGNED on the next line, so every
                # theta>0 run died with UnboundLocalError on its first request. Assign it first.
                qy = rowsB[k - 1:k - 1 + n_drafts].float().softmax(dim=-1).gather(
                    1, y_gpu.unsqueeze(1)).squeeze(1)
                _pt = st.get("ptil")
                if _pt is not None:
                    # the deployed gate, shared with the small-verifier configuration: p_tilde * q.
                    # With the proposer verifying itself both factors come from the same weights,
                    # so the product is that model's confidence measured twice -- under the block
                    # view and under exact conditioning. Keep the scale comparable by construction.
                    qy = qy * _pt
            mism = (qy < self.theta) if qy is not None else (y_gpu != g_gpu[:n_drafts])
            if bool(mism.any().item()):
                j = int(mism.int().argmax().item())             # first mismatch index
            else:
                j = n_drafts
                self._stats["full_runs"] += 1
            if j == 0:
                self._stats["seed_mismatch"] += 1               # g_1 != y_1: bf16 tie
            accepted = y_gpu[:j].tolist()
            t = int(g_gpu[j].item())
            self._stats["draft_match"] += j
            self._stats["draft_reject"] += (1 if j < n_drafts else 0)
            self._stats["cycles"] += 1

            emit = (list(rf) if self._emit_refeed.get(rpx, False) else []) + accepted + [t]
            next_token_ids_list.append(emit)
            mars_ws_list.append([1] * len(emit))

            self._emit_refeed[rpx] = False
            self._refeed[rpx] = [t]
            self._prefix_len[rpx] = self._prefix_len.get(rpx, 0) + len(emit)
            # realized contents of THIS window; on a full run t sits at S+blk, i.e. in the
            # next window, where it is written when it is re-fed
            dllm_tokens = (list(rf) + accepted + [t])[:blk]
            dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))
            self._dllm_write_override[rpx] = dllm_tokens
            self._advance_override[rpx] = k + j
            tc = n_drafts - j                       # rejected draft slots (incl. y_{j+1})
            trim_counts[bid] = tc
            sl = seq_lens_cpu[bid]
            for tt in range(tc):
                all_trim_rpx.append(rpx)
                all_trim_pos.append(sl - 1 - tt)
            self._stats["total_tokens"] += len(emit)

        # ---- batched KV-trim index lookup ----
        all_kv = req_to_token[all_trim_rpx, all_trim_pos] if all_trim_rpx else None
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
        self._tl_mark("exit"); self._tl_finish()
        return logits_output, next_token_ids_list, out.can_run_graph


Algorithm = MarsExactStream
