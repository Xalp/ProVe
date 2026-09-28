# Training a proposer

This directory reproduces the **proposer** of the paper: an ordinary instruction-tuned AR model
fine-tuned so that one forward pass proposes a whole block of 16 tokens. Everything here calls
the public [MARS](https://github.com/Xalp/MARS) repository unchanged; the scripts only pin its
options to the paper's recipe and remove the cluster plumbing.

If you only want to *run* ProVe, skip this directory: `inference/` serves the public 7B proposer
(see [Ready-made proposer](#ready-made-proposer)) with no training at all.

## What a proposer is

ProVe decodes in cycles. The **proposer** is a large AR model that proposes a *proposed block*
of B = 16 tokens in one forward pass: the input is extended with 16 placeholder tokens
(`<|mask|>`, id 151665, added to the vocabulary) and the model predicts the token at every
placeholder position given the context before the block. A **verifier** then reads the
proposed block in one more forward pass and commits the longest prefix it agrees with.

The proposer is trained with the MARS objective, which supervises both streams in one pass over
the concatenation `[placeholder stream | original sequence]`: the usual next-token loss on the
original tokens (`ar_weight 1.0`) plus the block loss at the placeholders, with a causal
attention pattern in which a placeholder sees the original tokens before its block and the
placeholders before it. Because the AR head keeps training, the proposer stays a normal AR
model: **ProVe-lossless** verifies the block with the proposer's own AR distribution and
reproduces greedy AR decoding token for token, and the AR head is also what `--config ar`
evaluates as the baseline.

## Recipe

Two stages, so that the incremental effect of the block objective can be measured cleanly
(stage 1 is the AR control; in the paper the block stage moves the six-benchmark AR average by
only -0.6 points relative to it).

| | Stage 1: AR SFT (`train/train_ar_sft.py`) | Stage 2: block training (`train/train_mars.py`) |
|---|---|---|
| Initialisation | `Qwen/Qwen2.5-32B-Instruct` | stage-1 `checkpoint-final` |
| Data | `allenai/Dolci-Instruct-SFT`, all 2,152,112 rows | seed-42 shuffled 12.5 % subset, 269,014 rows (`prepare_dolci_subset.py`) |
| Epochs | 1 | 1 |
| Learning rate | 5e-6, cosine, warmup ratio 0.03 | same |
| Sequence length | 512 (right truncation) | 512 per stream |
| Effective batch | 384 sequences | 384 |
| Block size | - | 16 (`--block_size 16 --right_shift_logits --ar_weight 1.0`) |
| Other | `group_by_length`, gradient checkpointing, bf16 | same, `--attn_implementation sdpa` |
| Parallelism | accelerate + DeepSpeed ZeRO-3, no offload | same |
| Compute | 150 GPU hours on 32 GPUs (80 GB class) | 384 GPU hours on 32 GPUs |

Per-device batch and gradient accumulation only change with the number of GPUs; the effective
batch is always 384 (12 x 1 x 32 and 6 x 2 x 32 in the paper; 12 x 4 x 8 and 6 x 8 x 8 on one
8-GPU node). Stage 2 uses half the per-device batch of stage 1 because every row carries both
streams. With more than one node, DeepSpeed's `zero_hpz_partition_size` is set to the GPUs per
node: parameters are sharded inside a node and replicated across nodes, so only the gradient
reduce-scatter crosses the network.

**Why a 12.5 % subset for stage 2.** Block training doubles every sequence and the block-causal
mask keeps attention on the slow path, so a stage-2 step costs roughly 17x a stage-1 step; the
full 2.15 M rows would take about 8 days on 32 GPUs. **Why it is shuffled before slicing.**
Dolci is stored ordered by source (its last ~200k rows are all Tool Use); a head slice would
silently drop whole domains. The seed-42 shuffle keeps every domain in proportion:

```
Coding 41,264   Reasoning 38,882   Chat 38,683   Math 33,727   Other 31,825   Tool Use 28,480
Precise IF 16,895   Safety 13,706   Science 13,045   Multilingual 12,495   Hardcoded Data 12
```

`prepare_dolci_subset.py` prints these counts; if the dataset changes upstream the row-count
assertion fails rather than silently producing a different subset.

## Setup

```bash
git clone https://github.com/Xalp/MARS.git && cd MARS && git checkout 131b6ae
pip install -r requirements.txt          # torch, transformers, accelerate, deepspeed, datasets, ...
export MARS_DIR=$PWD
```

MARS is used as a source tree, not a package: `train_proposer.sh` puts `$MARS_DIR` on
`PYTHONPATH` and runs `train/train_ar_sft.py` and `train/train_mars.py` from it with their public
options only (no patches). A recent CUDA build of PyTorch, DeepSpeed and `accelerate` are the
only hard requirements; `requirements.txt` pins the versions the authors used.

## Run

1. Build the stage-2 subset once (downloads Dolci on first use; a few minutes):

   ```bash
   python train/prepare_dolci_subset.py --out data/dolci_12p5
   ```

   It writes `data/dolci_12p5/train.parquet` (400 MB), which MARS reads with
   `--dataset_args data/dolci_12p5`.

2. Train on one node (8 GPUs, detected automatically):

   ```bash
   MARS_DIR=/path/to/MARS STAGE2_DATASET=data/dolci_12p5 OUT_DIR=runs/prove-32b \
     bash train/train_proposer.sh
   ```

   Stage 1 writes `runs/prove-32b/ar_sft/checkpoint-final`, stage 2
   `runs/prove-32b/mars_blk16/checkpoint-final` (the proposer). A checkpoint with optimizer
   state is saved every 300 steps (`save_only_model False`), so an interrupted stage resumes with

   ```bash
   STAGE=2 RESUME=runs/prove-32b/mars_blk16/checkpoint-2400 MARS_DIR=... STAGE2_DATASET=... OUT_DIR=runs/prove-32b \
     bash train/train_proposer.sh
   ```

   `STAGE=1` / `STAGE=2` run one stage; with `STAGE=all` a finished stage 1 is skipped.
   `STAGE2_INIT` starts stage 2 from another AR checkpoint (for example the public
   `Xalphinions/MARS-Qwen2.5-7B-AR-SFT`).

3. Multi-node (the paper: 4 nodes x 8 GPUs). Run the same command on every node with
   `NNODES=4 NODE_RANK=<0..3> MASTER_ADDR=<node-0 address> MASTER_PORT=29500`; all nodes need
   the same `MARS_DIR`, `OUT_DIR` and dataset paths on shared (or identical) storage.

**Memory.** ZeRO-3 over 8 GPUs keeps about 70 GB of parameters, gradients and fp32 optimizer
state per GPU for a 32B model, so on 80 GB GPUs a single node needs `OFFLOAD_OPTIMIZER=cpu`
(several times slower); 32 GPUs (the paper) or 8 GPUs with >=140 GB run without offload. A 7B
proposer trains comfortably on one 8 x 80 GB node: `BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
PER_DEVICE_BS_S1=24 PER_DEVICE_BS_S2=12` (the MARS 7B recipe; the script derives the
accumulation steps).

`bash train/train_proposer.sh --help` lists every variable. The DeepSpeed and accelerate
configurations are `configs/ds_zero3_hpz.json` and `configs/accelerate_zero3.yaml`; the script
copies them to `$OUT_DIR/configs/` and fills in the node-dependent fields, so the checked-in
files stay usable as they are for a single node.

## Ready-made proposer

The public MARS release `Xalphinions/MARS-Qwen2.5-7B-blk16` (Qwen2.5-7B-Instruct, 5 epochs of AR
SFT + 5 epochs of block training on all of Dolci, block size 16) is a valid proposer. Measured
with this repository (not reported in the paper, whose tables use the 32B proposer): on GSM8K,
ProVe-lossless decodes at 1.85x plain AR (433 vs 234 tok/s, 6.13 accepted tokens per cycle) at
75.1 vs 74.4 accuracy (`eval/run_eval.sh`, greedy, single request, 1024 new tokens).

```bash
hf download Xalphinions/MARS-Qwen2.5-7B-blk16 --local-dir models/MARS-Qwen2.5-7B-blk16
```

The 32B proposer of the paper is produced by the recipe above (`BASE_MODEL=Qwen/Qwen2.5-32B-Instruct`,
the default).

## Before serving: rewrite the checkpoint config

MARS saves checkpoints under its training-time model class: `config.json` says
`"architectures": ["A2DQwen2LMHeadModel"], "model_type": "a2d-qwen2"`. The inference stack serves
the very same weights as a plain Qwen2 model and needs
`"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"` (the placeholder token
`<|mask|>` is already inside the vocabulary; nothing else changes).

```bash
python inference/tools/prepare_checkpoint.py runs/prove-32b/mars_blk16/checkpoint-final models/prove-32b
bash inference/serve.sh --proposer models/prove-32b --tp 2
```

`prepare_checkpoint.py` writes a new directory with the rewritten `config.json` and links every
other file, so no weights are copied (`--copy` copies instead). `inference/serve.sh` runs it for
you when it sees an unconverted checkpoint, writing `<proposer>-qwen2` next to it (or the
directory given with `--prepared`). The same applies to the stage-1 AR checkpoint if you want
to serve it as the AR baseline.

## Notes

* On the DeepSpeed path the trainer's logged `loss` is the true loss multiplied by
  `gradient_accumulation_steps`; compare runs by that factor, not across GPU counts.
* `datasets` memory-maps its Arrow cache: keep `HF_DATASETS_CACHE` on a local disk, not on a
  network filesystem.
* The proposer must be served with the block size it was trained with (16). It does propose
  longer blocks when asked, but the speedup saturates from B = 24 on (paper appendix).
* `REPORT_TO=wandb` enables Weights & Biases logging (set `WANDB_API_KEY` in the environment).
