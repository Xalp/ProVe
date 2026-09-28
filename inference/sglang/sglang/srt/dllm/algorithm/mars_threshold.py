"""
MARS threshold-accept decoding (Option-4 rollout).

Adapted from LowConfidence, but with MARS's LEFT-TO-RIGHT acceptance instead of
confidence-anywhere reveal:

  Within each block, walk the masked positions left-to-right. Always accept the
  leftmost masked position (>=1 token per forward). Then keep accepting consecutive
  positions while their top-1 probability >= threshold (tau); stop at the first that
  falls below. This is exactly the MARS sliding-window accept gate.

Only the FIRST accepted position of each forward is "on-policy" (predicted from a
fully clean prefix); the RL trainer's Option-4 loss uses that (window-start) flag.
This server just needs to emit the accepted tokens + (optionally) which were the
first-of-forward; window-start can also be reconstructed from the per-forward accept
counts on the trainer side.

Config keys (--dllm-algorithm-config YAML):
  threshold: float, top-1 prob acceptance gate tau (e.g. 0.95). Default 0.95.

--------------------------------------------------------------------------------
DEVICE<->HOST SYNC DISCIPLINE (why this file looks vectorised)

Every scalar readback (`.item()`, `.tolist()`, `bool(t)`, `t.nonzero()`) and every
BLOCKING host->device copy (`t[i] = python_int`, `torch.tensor(list, device=cuda)`)
drains the CUDA stream: the CPU waits for the GPU to finish, then the GPU idles
while Python gets to the next launch. The original scalar implementation did
O(batch_size * block_size) of them per cycle — a per-position confidence loop
inside `_accept_mask_ltr`, a per-position `.item()` audit loop, a per-block
`torch.sum(...).item()` per forward, a `req_pool_indices[b].item()` per block per
forward. On a 4B/H100 that cost ~7.8 ms per MODEL FORWARD (16.9 -> 9.1 ms at
TP=1, 15.7 -> 7.4 ms at TP=2, marsT95 tau .95 block 16, batch 1).

The rule here: the accept gate, the audit and the bookkeeping all run as dense
[num_windows, block_size] tensor ops, and the host reads the device exactly

    1x for the setup, 1x per model forward (the per-block masks-left counts),
    1x for the commit forward (pending seeds + audit mismatches + window-start
    bits)

Counted on a real block-16 cycle (by patching Tensor.item/tolist/nonzero):
171 -> 17 readbacks per cycle at batch 1, 2469 -> 17 at batch 16 -- the new
count does not grow with the batch. Host->device transfers go through
`_h2d` (pinned staging + non_blocking) so they are async too. Semantics are
UNCHANGED: `_accept_mask_ltr` below is kept as the scalar definition of the gate
and the vectorised path reproduces it bit-for-bit (the tau comparison is done in
float64 precisely because the scalar path compared a python float against
`float(tensor.item())`).
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

# Peak activation budget for the row-chunked softmax that materialises top-1
# probabilities. Softmax is row-wise, so chunking cannot change a single value;
# it only bounds rows*vocab*itemsize.
_PROB_CHUNK_BYTES = 64 << 20
# Distinct (batch layout) index-tensor sets kept alive. Steady-state decode has
# exactly one; prefill batches churn keys, hence the cap.
_LAYOUT_CACHE_MAX = 64


class MarsThreshold(DllmAlgorithm):

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.allow_per_request_tau = config.algorithm_config.get(
            "allow_per_request_tau", True
        )
        # Structure-aware tau (ported from idlm_blockN): force AR (tau>1 -> only the
        # window-start token commits) inside <tool_call>...</tool_call> spans.
        self.code_aware_tau = config.algorithm_config.get("code_aware_tau", False)
        self.code_open_token = int(config.algorithm_config.get("code_open_token", 151657))
        self.code_close_token = int(config.algorithm_config.get("code_close_token", 151658))
        self.code_force_tau = float(config.algorithm_config.get("code_force_tau", 2.0))
        self._in_code = {}  # req_pool_idx -> bool
        # Emitted-token sampling temperature (HF generate_mars semantics: the accept
        # GATE always uses the intrinsic top-1 prob; only the EMITTED token is sampled).
        self.temperature = float(config.algorithm_config.get("temperature", 0.0))
        # Pending protocol (mirrors IDLMBlockN): the pure-decode fast path hands every
        # block over ALL-MASK ("the algorithm's classify phase overwrites non-mask
        # positions") — but this algorithm predates that contract and assumed start>=1.
        # Without a seed, roll-by-1 predicts block[0] from the WRAPPED logit of the last
        # mask row => one corrupted token at every block boundary, and the token implied
        # by the prefill logits was silently dropped. _pending carries that next token
        # (exact AR: sampled from logits whose inputs were all real) across rounds.
        self._pending = {}
        # (layout key) -> device index tensors; rebuilt only when the batch shape changes.
        self._layout_cache = {}

    def cleanup_request(self, req_pool_idx: int):
        self._in_code.pop(req_pool_idx, None)
        self._pending.pop(req_pool_idx, None)
        import os as _os
        if _os.environ.get("MARSV_LOG"):
            import logging as _logging
            s = self._stats
            ap, am = s.get("audit_pairs", 0), s.get("audit_mismatch", 0)
            _logging.getLogger(__name__).info(
                f"[MarsThreshold] cum stats: fwd={s['total_forwards']} "
                f"tok={s['total_tokens']} "
                f"tok/fwd={s['total_tokens']/max(1,s['total_forwards']):.3f} "
                f"audit_false_accept={am}/{ap} ({am/max(1,ap):.4f})")

    # ------------------------------------------------------------------
    # host <-> device plumbing (see the module docstring)
    # ------------------------------------------------------------------
    @staticmethod
    def _h2d(values, device, dtype=torch.int64) -> torch.Tensor:
        """CPU list -> device tensor WITHOUT draining the stream.

        `torch.tensor(v, device=cuda)` issues a blocking cudaMemcpy (a full
        stream sync). Staging through PyTorch's caching *pinned* host allocator
        and copying with non_blocking=True keeps it asynchronous; the allocator
        records a stream event on the pinned block so it is not recycled before
        the copy lands.
        """
        try:
            cpu = torch.tensor(values, dtype=dtype, pin_memory=True)
            return cpu.to(device, non_blocking=True)
        except Exception:
            return torch.tensor(values, dtype=dtype, device=device)

    def _cache_put(self, key, entry):
        if len(self._layout_cache) >= _LAYOUT_CACHE_MAX:
            self._layout_cache.clear()
        self._layout_cache[key] = entry
        return entry

    def _seg_index(self, device, lens: List[int], n_rows: int):
        """Per-segment index tensors for a given ragged layout (cached).

        offs_t   [bs+1] segment boundaries (row space)
        tail_pos [bs, block_size] the last block_size rows of each segment
        last_idx [bs]  the last row of each segment
        """
        key = ("seg", device, n_rows, tuple(lens))
        e = self._layout_cache.get(key)
        if e is not None:
            return e
        offs = [0]
        for ln in lens:
            offs.append(offs[-1] + ln)
        hi = max(n_rows, 1)
        offs_t = self._h2d(offs, device)
        ar = torch.arange(self.block_size, device=device, dtype=torch.int64)
        base = (offs_t[1:] - self.block_size).clamp_min(0)
        tail_pos = (base[:, None] + ar[None, :]).clamp_(0, hi - 1)
        return self._cache_put(key, {
            "offs": offs,
            "offs_idx": offs_t.clamp(0, n_rows),   # safe gather into cumsum-with-0-prefix
            "tail_pos": tail_pos,
            "last_idx": (offs_t[1:] - 1).clamp_(0, hi - 1),
        })

    def _win_index(self, device, lens: List[int], n_rows: int, is_window: List[bool]):
        """Row indices of the decode WINDOWS only: [W, block_size], plus the
        contiguous logits span that covers them."""
        key = ("win", device, n_rows, tuple(lens), tuple(is_window))
        e = self._layout_cache.get(key)
        if e is not None:
            return e
        seg = self._seg_index(device, lens, n_rows)
        offs = seg["offs"]
        wi = [b for b in range(len(lens)) if is_window[b]]
        if len(wi) == len(lens):
            win_pos = seg["tail_pos"]
        elif wi:
            win_pos = seg["tail_pos"].index_select(0, self._h2d(wi, device))
        else:
            win_pos = seg["tail_pos"][:0]
        lo = max(0, min(offs[b + 1] - self.block_size for b in wi)) if wi else 0
        hi = max(offs[b + 1] for b in wi) if wi else 0
        return self._cache_put(key, {
            "wi": wi,
            "win_pos": win_pos,
            "rel": win_pos - lo,     # index into logits[lo:hi]
            "lo": lo,
            "hi": hi,
        })

    def _top1_and_prob(self, logits: torch.Tensor):
        """Row-wise argmax and the softmax probability of that argmax.

        Identical, value for value, to the original per-block
        `softmax(rolled) -> gather(argmax(rolled))`: softmax and argmax are
        row-wise, so neither the row chunking nor the missing roll (which only
        permutes rows) can move a number. Rolling is applied afterwards to the
        two [W, B] results instead of to a [B, vocab] logits copy.
        """
        n, vocab = logits.shape[0], logits.shape[-1]
        top1 = torch.argmax(logits, dim=-1)
        p = torch.empty(n, dtype=logits.dtype, device=logits.device)
        step = max(1, _PROB_CHUNK_BYTES // max(1, vocab * logits.element_size()))
        for s in range(0, n, step):
            e = min(n, s + step)
            probs = F.softmax(logits[s:e], dim=-1)
            p[s:e] = torch.gather(probs, -1, top1[s:e].unsqueeze(-1)).squeeze(-1)
        return top1, p

    def _next_from_row(self, row_logits: torch.Tensor) -> int:
        if self.temperature > 0:
            return int(torch.multinomial(
                F.softmax(row_logits / self.temperature, dim=-1), 1).item())
        return int(torch.argmax(row_logits).item())

    def _next_from_rows(self, logits: torch.Tensor, rows: List[int]) -> List[int]:
        """Batched `_next_from_row`: ONE readback for the whole batch.

        The temperature>0 branch keeps the per-row multinomial draws (and their
        order) so the RNG stream is untouched.
        """
        if not rows:
            return []
        sub = logits.index_select(0, self._h2d(rows, logits.device))
        if self.temperature > 0:
            return [self._next_from_row(sub[i]) for i in range(len(rows))]
        return torch.argmax(sub, dim=-1).tolist()

    def _update_in_code(self, rpx, tokens):
        if not self.code_aware_tau or rpx is None:
            return
        for t in tokens:
            t = int(t)
            if t == self.code_open_token:
                self._in_code[rpx] = True
            elif t == self.code_close_token:
                self._in_code[rpx] = False

    def _tau_for_bid(self, forward_batch, bid: int) -> float:
        """Per-request tau from sampling_params.custom_params["mars_tau"]
        (requires --enable-custom-logit-processor so custom_params reach
        SamplingBatchInfo), else the global default."""
        if not self.allow_per_request_tau:
            return self.threshold
        si = getattr(forward_batch, "sampling_info", None)
        cps = getattr(si, "custom_params", None) if si is not None else None
        if cps is not None and bid < len(cps) and cps[bid]:
            tau = cps[bid].get("mars_tau")
            if tau is not None:
                return float(tau)
        return self.threshold

    def _accept_mask_ltr(self, block_input_ids, confidence, threshold):
        """Left-to-right MARS accept mask for one block. SCALAR REFERENCE.

        Not used by run() any more (it costs one device sync per position); it
        stays as the executable definition of the gate that the vectorised
        `_accept_mask_batched` must reproduce exactly.

        block_input_ids: [block_size] (mask_id at undecided positions)
        confidence:      [block_size] top-1 prob at each position (-inf at non-mask)
        Returns a bool tensor [block_size]: positions to commit this forward.
        """
        masked = block_input_ids == self.mask_id
        accept = torch.zeros_like(masked)
        masked_idx = masked.nonzero(as_tuple=True)[0]  # ascending = left-to-right
        for k, pos in enumerate(masked_idx.tolist()):
            if k == 0:
                accept[pos] = True  # always commit the leftmost (>=1 per forward)
            elif float(confidence[pos].item()) >= threshold:
                accept[pos] = True
            else:
                break
        return accept

    @staticmethod
    def _accept_mask_batched(masked: torch.Tensor, conf: torch.Tensor,
                             taus: torch.Tensor):
        """Vectorised `_accept_mask_ltr` over [W, block_size], no device syncs.

        masked: bool [W, B]              conf: top-1 prob [W, B] (any float dtype)
        taus:   float64 [W, 1]           -> (accept bool [W, B], first bool [W, B])

        `first` marks the leftmost masked position of each row (always accepted,
        and exactly the Option-4 window-start bit). A row accepts the maximal
        prefix of its masked positions whose confidence clears tau, stopping at
        the first failure — expressed as "no earlier masked non-first position
        failed", i.e. a prefix count of failures equal to zero.

        The comparison runs in float64 because the scalar path compared a python
        float (double) against `float(conf[pos].item())`; comparing in float32
        would silently round tau and could flip a boundary case. It is written as
        `~(conf >= tau)` rather than `conf < tau` so a NaN confidence rejects
        (the scalar `nan >= tau` is False -> break), not accepts.
        """
        csum = torch.cumsum(masked.to(torch.int32), dim=1)
        first = masked & (csum == 1)
        fail = masked & (~first) & (~(conf.to(torch.float64) >= taus))
        accept = masked & (torch.cumsum(fail.to(torch.int32), dim=1) == 0)
        return accept, first

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        B = self.block_size
        num_forwards = 0
        start_list = []

        input_ids = forward_batch.input_ids
        device = input_ids.device
        n_rows = input_ids.numel()
        mask_index = input_ids == self.mask_id

        # Ragged segment layout: the inline-absorb path merges fresh requests'
        # FULL PROMPTS into the decode batch (prepare_for_dllm_decode), so
        # per-request extents come from extend_seq_lens — never assume a
        # uniform (batch_size, block_size) grid. A segment WITH mask tokens is
        # a decode window (block_size, at the segment tail); a segment WITHOUT
        # is an inline prefill: KV-write + pending seed only, no emission.
        _el = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if _el is None:
            _el = getattr(forward_batch, "extend_seq_lens", None)
            if isinstance(_el, torch.Tensor):
                _el = _el.tolist()
        lens_raw = list(_el) if _el is not None else None
        lens = list(_el) if _el else [B] * batch_size
        if sum(lens) != n_rows:
            lens = [B] * batch_size
        seg = self._seg_index(device, lens, n_rows)
        offs = seg["offs"]

        # ---- ONE readback for the whole setup ------------------------------
        # total mask count (prefill fast path), per-segment mask count
        # (is_window), per-tail-block mask count (start), req_pool_indices.
        cum = torch.cumsum(mask_index.to(torch.int64), dim=0)
        cumpad = torch.cat([cum.new_zeros(1), cum])            # cumpad[i] = masks in [0,i)
        seg_counts = cumpad[seg["offs_idx"][1:]] - cumpad[seg["offs_idx"][:-1]]
        blk_counts = mask_index[seg["tail_pos"]].sum(dim=1)
        rp = getattr(forward_batch, "req_pool_indices", None)
        if isinstance(rp, torch.Tensor) and rp.numel() >= batch_size:
            rp64 = rp[:batch_size].to(torch.int64)
        else:
            rp64 = torch.full((batch_size,), -1, dtype=torch.int64, device=device)
        head = torch.cat([cumpad[-1:], seg_counts, blk_counts, rp64]).tolist()
        total_masked = int(head[0])
        seg_counts_c = head[1:1 + batch_size]
        blk_counts_c = head[1 + batch_size:1 + 2 * batch_size]
        rpx_list = [None if int(v) < 0 else int(v)
                    for v in head[1 + 2 * batch_size:1 + 3 * batch_size]]

        # Fast path: no mask tokens -> plain forward (prefill), save kv cache.
        # Capture the next-token seed per request from the chunk's LAST row: chunked
        # prefill overwrites it each chunk, so the final chunk's (true) value wins.
        if total_masked == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            self._stats["total_forwards"] += 1
            fl = getattr(out.logits_output, "full_logits", None)
            if fl is not None and lens_raw is not None:
                rows, bids, off = [], [], 0
                for bid, ln in enumerate(lens_raw):
                    if bid < batch_size and rpx_list[bid] is not None and ln > 0:
                        rows.append(off + ln - 1)
                        bids.append(bid)
                    off += ln
                for bid, val in zip(bids, self._next_from_rows(fl, rows)):
                    self._pending[rpx_list[bid]] = val
            return out.logits_output, [], out.can_run_graph

        is_window = [c > 0 for c in seg_counts_c]
        wnd0 = [offs[b + 1] - B for b in range(batch_size)]

        base_taus = [self._tau_for_bid(forward_batch, bid) for bid in range(batch_size)]
        # sync code-state from tokens already committed in each segment
        if self.code_aware_tau:
            all_ids = input_ids.tolist()          # one readback, only in this mode
            for bid in range(batch_size):
                self._update_in_code(
                    rpx_list[bid],
                    [t for t in all_ids[offs[bid]:offs[bid + 1]] if t != self.mask_id],
                )

        # Seed pure-decode blocks with their pending next token, batched into a
        # single async host->device scatter (a per-block `blk[0] = int` would be
        # one blocking copy per block per cycle).
        seed_pos, seed_val, blk_rem = [], [], []
        for block_id in range(batch_size):
            rpx = rpx_list[block_id]
            if not is_window[block_id]:
                # Inline prefill segment: emits nothing this round.
                start_list.append(B)
                continue
            start = B - int(blk_counts_c[block_id])
            blk_rem.append(int(blk_counts_c[block_id]))
            if start == 0:
                # Pure-decode template: block arrived all-mask. Seed block[0] with the
                # pending next token (exact AR). start stays 0 so it is EMITTED.
                pend = self._pending.pop(rpx, None) if rpx is not None else None
                if pend is not None:
                    seed_pos.append(max(0, wnd0[block_id]))
                    seed_val.append(int(pend))
                    if int(pend) != self.mask_id:
                        blk_rem[-1] -= 1
            else:
                # Slow path handed us a committed prefix (already-emitted tokens);
                # any stale pending is already part of it.
                if rpx is not None:
                    self._pending.pop(rpx, None)
            start_list.append(start)
        if seed_pos:
            input_ids[self._h2d(seed_pos, device)] = self._h2d(
                seed_val, device, dtype=input_ids.dtype)

        win = self._win_index(device, lens, n_rows, is_window)
        wi, win_pos, rel = win["wi"], win["win_pos"], win["rel"]
        W = len(wi)
        col = torch.arange(B, device=device, dtype=torch.int64).unsqueeze(0)
        taus_t = self._h2d([base_taus[b] for b in wi], device,
                           dtype=torch.float64).unsqueeze(1) if W else None
        # MARS Option-4 bookkeeping: per emitted position, 1 iff it was the
        # FIRST token accepted in its forward (clean-prefix conditioned, hence
        # on-policy at any tau). Shipped via customized_info["mars_ws"].
        ws_gpu = torch.zeros((W, B), dtype=torch.bool, device=device)

        # Iterate: each forward commits a left-to-right run per block. `blk_rem`
        # mirrors, on the host, the per-block mask count the scalar path read back
        # with `torch.sum(block_mask_index).item()` at the top of every block loop.
        for _ in range(B):
            if sum(blk_rem) <= 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            num_forwards += 1

            input_ids = forward_batch.input_ids
            sub = logits_output.full_logits[win["lo"]:win["hi"]]
            # MARS right_shift_logits convention: the model's raw logit at position
            # p-1 predicts token p (standard AR). So to predict the masked token at
            # block index j, use the logit at j-1. The block always begins with >=1
            # committed token (start>=1), so roll-by-1 is valid for all mask positions
            # (index 0 is committed -> its wrapped logit is never used).
            top1_all, p_all = self._top1_and_prob(sub)
            top1 = torch.roll(top1_all[rel], shifts=1, dims=1)   # [W, B]
            conf = torch.roll(p_all[rel], shifts=1, dims=1)      # gate confidence
            blk = input_ids[win_pos]
            masked = blk == self.mask_id

            # emitted token: temperature-sampled (diverse rollouts), else argmax
            x = top1.to(blk.dtype)
            if self.temperature > 0:
                # One draw per STILL-UNFINISHED block, in ascending block order:
                # byte-for-byte the RNG stream the scalar path consumed (it
                # `continue`d past blocks with no masks left before sampling).
                act = [w for w in range(W) if blk_rem[w] > 0]
                if act:
                    drawn = torch.stack([
                        torch.multinomial(
                            F.softmax(
                                torch.roll(
                                    logits_output.full_logits[
                                        wnd0[wi[w]]:offs[wi[w] + 1]],
                                    shifts=1, dims=0) / self.temperature, dim=-1),
                            1).squeeze(-1)
                        for w in act
                    ]).to(blk.dtype)
                    x = x.index_copy(0, self._h2d(act, device), drawn)

            if self.code_aware_tau:
                forced = [self.code_force_tau
                          if (rpx_list[b] is not None and self._in_code.get(rpx_list[b], False))
                          else base_taus[b] for b in wi]
                taus_t = self._h2d(forced, device, dtype=torch.float64).unsqueeze(1)

            accept, first = self._accept_mask_batched(masked, conf, taus_t)
            new_ids = torch.where(accept, x, blk)
            input_ids[win_pos] = new_ids
            ws_gpu |= first

            rem_t = (new_ids == self.mask_id).sum(dim=1)      # [W]
            if self.code_aware_tau:
                pay = torch.cat([rem_t,
                                 accept.reshape(-1).to(torch.int64),
                                 new_ids.reshape(-1).to(torch.int64)]).tolist()
                blk_rem = [int(v) for v in pay[:W]]
                acc_c, tok_c = pay[W:W + W * B], pay[W + W * B:]
                for w, b in enumerate(wi):
                    self._update_in_code(
                        rpx_list[b],
                        [tok_c[w * B + j] for j in range(B) if acc_c[w * B + j]])
            else:
                blk_rem = rem_t.tolist()        # the loop's only readback

        # Commit forward: refresh KV cache with the finalized tokens. Its last-row
        # logits are the exact-AR prediction for the NEXT block's first token.
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        num_forwards += 1
        input_ids = forward_batch.input_ids
        _fl = getattr(logits_output, "full_logits", None)
        ws_flat = None
        if _fl is not None:
            # Free audit (same as MarsVerifier): commit-forward inputs are all real, so
            # its logits are exact AR — count emitted tokens that AR would not have said.
            self._stats.setdefault("audit_pairs", 0)
            self._stats.setdefault("audit_mismatch", 0)
            pairs = 0
            ar_last = None
            if W:
                ar_sub = torch.argmax(_fl[win["lo"]:win["hi"]], dim=-1)[rel]  # [W, B]
                ar_top1 = torch.roll(ar_sub, shifts=1, dims=1)
                starts_t = self._h2d([max(1, start_list[b]) for b in wi],
                                     device).unsqueeze(1)
                mism = ((ar_top1 != input_ids[win_pos]) & (col >= starts_t)).sum().view(1)
                for b in wi:
                    pairs += B - max(1, start_list[b])
                if W == batch_size:
                    ar_last = ar_sub[:, B - 1]   # last row of every segment
            else:
                mism = torch.zeros(1, dtype=torch.int64, device=device)

            # Pending seed from each segment's LAST row: for a decode window that
            # is the next block's first token; for an inline prefill segment it is
            # the request's first generated token (exact AR from the full prompt).
            if self.temperature > 0:
                pend_vals = self._next_from_rows(
                    _fl, [offs[b + 1] - 1 for b in range(batch_size)])
                tail = torch.cat(
                    [mism.to(torch.int64), ws_gpu.reshape(-1).to(torch.int64)]).tolist()
                mism_c, ws_flat = int(tail[0]), tail[1:]
            else:
                if ar_last is None:
                    ar_last = torch.argmax(
                        _fl.index_select(0, seg["last_idx"]), dim=-1)
                tail = torch.cat([ar_last.to(torch.int64),
                                  mism.to(torch.int64),
                                  ws_gpu.reshape(-1).to(torch.int64)]).tolist()
                pend_vals = [int(v) for v in tail[:batch_size]]
                mism_c, ws_flat = int(tail[batch_size]), tail[batch_size + 1:]
            for bid in range(batch_size):
                if rpx_list[bid] is not None:
                    self._pending[rpx_list[bid]] = pend_vals[bid]
            self._stats["audit_pairs"] += pairs
            self._stats["audit_mismatch"] += mism_c
        elif W:
            ws_flat = ws_gpu.reshape(-1).tolist()

        ws_per_block = [[0] * B for _ in range(batch_size)]
        if ws_flat is not None:
            for w, b in enumerate(wi):
                ws_per_block[b] = [int(v) for v in ws_flat[w * B:(w + 1) * B]]

        next_token_ids_list = [
            input_ids[wnd0[i] + start_list[i]:offs[i + 1]]
            if is_window[i] else input_ids[:0]
            for i in range(batch_size)
        ]

        import os as _os
        if _os.environ.get("TAU_CYCLE_LOG"):
            self._cyc = getattr(self, "_cyc", 0) + 1
            if self._cyc <= 60 or self._cyc % 200 == 0:
                _mask_emit = [int((t == self.mask_id).sum().item())
                              for t in next_token_ids_list]
                print(
                    f"[tau cycle {self._cyc}] bs={batch_size} lens={lens} "
                    f"windows={is_window} starts={start_list} "
                    f"emits={[len(t) for t in next_token_ids_list]} "
                    f"masks_in_emit={_mask_emit} nfwd={num_forwards} "
                    f"rpxs={rpx_list}",
                    flush=True)

        total_new_tokens = sum(B - start_list[i] for i in range(batch_size))
        self._stats["total_forwards"] += num_forwards
        self._stats["total_tokens"] += total_new_tokens

        # Window-start bits for the emitted slice of each block.
        logits_output.customized_info = {
            "mars_ws": [ws_per_block[i][start_list[i]:] for i in range(batch_size)]
        }

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = MarsThreshold
