# ProVe changes to the bundled sglang fork

Base: the sglang fork bundled in I-DLM (https://github.com/Introspective-Diffusion/I-DLM,
commit `a23c1a1`, directory `inference/sglang/`). Paths below are relative to `sglang/srt/`.

## New files (ProVe)

| file | role |
|---|---|
| `dllm/algorithm/prove.py` | `ProVe` — the paper's name; subclass/alias of `MarsVerifierStream` |
| `dllm/algorithm/prove_lossless.py` | `ProVeLossless` — the paper's name; subclass/alias of `MarsExactStream` |
| `dllm/algorithm/mars_exact_stream.py` | ProVe-lossless: the proposer verifies its own proposed block (two proposer forwards per cycle); output equals greedy AR decoding |
| `dllm/algorithm/mars_verifier_stream.py` | ProVe: one proposer forward + one verifier forward per cycle; gate `p~ * q >= theta` |
| `dllm/algorithm/mars_verifier.py` | `_FastVerifier` / `_GraphVerifier`: HF-based verifier steppers, the fallback chain when the in-process verifier cannot be built |
| `dllm/inproc_verifier.py` | the in-process verifier: a second sglang `ModelRunner` (CUDA graphs, own KV pool) next to the proposer — the configuration used in the paper |
| `dllm/algorithm/mars_threshold.py` | `MarsThreshold`: the MARS self-gate baseline (accept while the proposer's own top-1 probability >= tau) |

## Modified I-DLM / sglang files (14, +482 / -21 lines)

| file | what changed |
|---|---|
| `dllm/algorithm/idlm_blockN.py` | I-DLM's own algorithm, kept working. Adds a per-request tau (`sampling_params.custom_params["mars_tau"]`), a structure-aware force-AR mode inside tool-call spans, the MARS tau gate on the non-spec-verify path (also for temperature > 0), per-round window-start bits in `customized_info["mars_ws"]`, and timing/debug output gated behind `IDLM_TIMING` / `IDLM_STEP_DEBUG` / `IDLM_ACCEPT_LOG`. |
| `dllm/config.py` | `Qwen2ForCausalLM` and `Qwen2_5_VLForConditionalGeneration` entries (mask token `<\|mask\|>` = 151665); `mask_id` can be overridden from the yaml. |
| `layers/attention/flashinfer_backend.py` | Opt-in `DLLM_CUSTOM_MASK=1`: explicit boolean window masks for dLLM window forwards (needed only by tree-shaped proposals, not by the shipped configurations). Fix: pass exactly the used `paged_kv_indices` entries to `plan()` (a grown buffer crashed long-context window replays). |
| `managers/mm_utils.py` | Multi-token decode windows lie past the prompt and hold no multimodal placeholders: skip the scatter instead of raising, and clamp out-of-vocab ids before the embedding gather. |
| `managers/scheduler.py` | Honour the algorithm's KV-trim / advance / write overrides on an empty emission; collect per-round `customized_info` on the fast path; flush a pending deferred stream batch before overwriting it (a finished request's final response was otherwise dropped and the client hung). |
| `managers/scheduler_output_processor_mixin.py` | List-valued `customized_info` entries are per-token chunks: extend instead of append so multi-token rounds stay aligned with `send_token_offset`. |
| `managers/tokenizer_manager.py` | Accumulate `customized_info` across streamed chunks instead of overwriting; re-export three request structs from `io_struct` (compatibility shim for RL trainers). |
| `managers/tp_worker.py` | Build the in-process verifier eagerly right after the algorithm is constructed (for `ProVe` / `MarsVerifierStream`): its CUDA graphs must be captured before the proposer's graphs ever replay. |
| `model_executor/cuda_graph_runner.py` | `DLLM_GRAPH_DEBUG=1` diagnostic: log why a dLLM forward is (not) eligible for graph replay. |
| `model_executor/forward_batch_info.py` | Extend mrope positions past the prompt for multi-token decode windows (Qwen2-VL proposers). |
| `models/qwen2.py` | `LogitsProcessor(config, return_full_logits=True)` so block decoding gets per-position logits (harmless for the AR path). |
| `models/qwen2_5_vl.py` | Same `return_full_logits=True` change. |
| `multimodal/processors/qwen_vl.py` | Accept prompts whose vision placeholders are already expanded to tokens. |
| `sampling/sampling_batch_info.py` | Keep `custom_params` even without a custom logit processor (per-request tau) and keep them aligned through `filter_batch` / `merge_batch`. |

## Dropped from the release (experimental; nothing shipped imports them)

| file | why |
|---|---|
| `dllm/algorithm/mars_anchor_stream.py` | experimental configuration (small-model continuation of the proposed block); not in the paper |
| `dllm/algorithm/mars_cross_stream.py` | experimental configuration (small block-proposer, large verifier); not in the paper |
| `dllm/algorithm/mars_tree_stream.py` | experimental block-tree self-verification; not in the paper |
| `dllm/algorithm/block_tree.py` | helper used only by `mars_tree_stream.py` |
| `dllm/sgl_verifier.py`, `dllm/verifier_worker.py` | an opt-in subprocess verifier (`sgl_verifier: true`), superseded by `inproc_verifier.py`; it carried machine-specific paths and its numerics were never validated. Setting `sgl_verifier: true` now raises a `ValueError`. |

Kept although ProVe does not use them (I-DLM upstream, tracked in its git):
`dllm/algorithm/joint_threshold.py`, `dllm/algorithm/low_confidence.py`,
`dllm/algorithm/fused_verify_kernel.py`.

Junk never shipped: `*.bak_*`, `*.prelogpatch`, `*.preoverlap`, `__pycache__/`, `*.pyc`.

## Release edits (this repository vs. the internal tree)

* `managers/tp_worker.py`: the class-name tuple that triggers the eager in-process verifier
  build is now `("ProVe", "MarsVerifierStream")`.
* `dllm/algorithm/mars_verifier_stream.py`: `verifier_path` has no default any more (a clear
  `ValueError` asks for it); the `sgl_verifier` / `sgl_graph` / `sgl_blk32_yaml` keys and
  the subprocess branch of `_load_verifier` were removed.
* `dllm/algorithm/mars_verifier_stream.py`: a `verifier_path` that is empty or still the
  placeholder `__VERIFIER__` raises the same `ValueError`; `inproc_mem_fraction` no longer
  defaults to 0.20 on the algorithm object (`None` = let `inproc_pool_gb` size the pool) and
  `inproc_pool_gb` is mirrored as an attribute.
* `dllm/inproc_verifier.py`: a pinned `inproc_mem_fraction` whose predicted pool is
  `<= 0` raises at start-up with the arithmetic instead of letting sglang fail four frames
  down; `maybe_build` re-raises a failed build (the server stops with the reason) unless
  `inproc_fallback: true` is in the yaml or `MARSV_INPROC_FALLBACK=1` is set, which restores
  the old fall-back to the transformers verifier with a WARNING; the unknown-checkpoint
  weight guess is 6.5 GB (a 3B) instead of 1.5 GB.
* `dllm/algorithm/mars_verifier.py`: the default `verifier_path` of the `MarsVerifier`
  variant is `Qwen/Qwen2.5-3B-Instruct`.
* Docstrings and comments use the paper's terms (proposer / verifier / proposed block) and
  sizes; internal-experiment remarks were removed. Variable names are unchanged.
* Added `dllm/algorithm/prove.py`, `dllm/algorithm/prove_lossless.py`, this file,
  `README.md` and `LICENSE` (Apache-2.0, SGLang).

The algorithm registry (`dllm/algorithm/__init__.py`, I-DLM) imports every module of the
package and registers `module.Algorithm` under its class name, so the names accepted by
`--dllm-algorithm` are: `ProVe`, `ProVeLossless`, `MarsVerifierStream`, `MarsExactStream`,
`MarsThreshold`, `MarsVerifier` (an earlier block-fill variant kept for its helper classes),
plus I-DLM's `IDLMBlockN`, `JointThreshold`, `LowConfidence`.
