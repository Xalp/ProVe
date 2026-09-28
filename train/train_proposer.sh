#!/usr/bin/env bash
# ======================================================================================
# train_proposer.sh -- train a ProVe proposer with the MARS recipe, in two stages.
#
#   stage 1  AR SFT           train/train_ar_sft.py   of the public MARS repo (commit 131b6ae)
#   stage 2  block training   train/train_mars.py     block_size 16, from the stage-1 checkpoint
#
# Usage
#   MARS_DIR=/path/to/MARS bash train/train_proposer.sh          # the paper's recipe: 1 + 1 epochs on Dolci
#   bash train/train_proposer.sh --help
#
# Everything is an environment variable (defaults in brackets):
#   MARS_DIR          clone of https://github.com/Xalp/MARS checked out at 131b6ae      [required]
#   BASE_MODEL        [Qwen/Qwen2.5-32B-Instruct]
#   DATASET           stage-1 data                                        [allenai/Dolci-Instruct-SFT]
#   STAGE2_DATASET    stage-2 data                                        [$DATASET, i.e. the same rows again]
#                     (a directory written by prepare_dolci_subset.py gives the cheaper 12.5 % variant)
#   STAGE2_INIT       checkpoint stage 2 starts from                      [$OUT_DIR/ar_sft/checkpoint-final]
#   OUT_DIR           [runs/prove-proposer]  -> $OUT_DIR/ar_sft and $OUT_DIR/mars_blk$BLOCK_SIZE
#   BLOCK_SIZE [16]   EPOCHS [1]   LR [5e-6]   MAX_LENGTH [512]   EFF_BATCH [384]
#   PER_DEVICE_BS_S1 [12]  PER_DEVICE_BS_S2 [6]   gradient accumulation is derived so that the
#                                                  effective batch is EFF_BATCH in both stages
#   NNODES [1]  NODE_RANK [0]  MASTER_ADDR [127.0.0.1]  MASTER_PORT [29500]
#   GPUS_PER_NODE     [detected with nvidia-smi, else 8]
#   STAGE             all | 1 | 2                                          [all]
#   RESUME            checkpoint dir passed as --resume_from_checkpoint to the stage that launches
#   OFFLOAD_OPTIMIZER none | cpu  (cpu lets a 32B run fit on one 8x80GB node, several times slower) [none]
#   REPORT_TO         none | wandb                                         [none]
#   NUM_PROC          dataset preprocessing workers                        [min(64, nproc)]
#
# Multi-node: run the same command on every node with NNODES, NODE_RANK, MASTER_ADDR set;
# every node needs the same MARS_DIR / OUT_DIR / dataset paths (shared or identical storage).
# ======================================================================================
set -uo pipefail
case "${1:-}" in -h|--help) sed -n '3,/^# =====/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0;; esac

HERE=$(cd "$(dirname "$0")" && pwd)
MARS_DIR=${MARS_DIR:?set MARS_DIR to a clone of https://github.com/Xalp/MARS (commit 131b6ae)}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-32B-Instruct}
DATASET=${DATASET:-allenai/Dolci-Instruct-SFT}
STAGE2_DATASET=${STAGE2_DATASET:-$DATASET}
OUT_DIR=${OUT_DIR:-runs/prove-proposer}
BLOCK_SIZE=${BLOCK_SIZE:-16}
EPOCHS=${EPOCHS:-1}
LR=${LR:-5e-6}
MAX_LENGTH=${MAX_LENGTH:-512}
EFF_BATCH=${EFF_BATCH:-384}
PER_DEVICE_BS_S1=${PER_DEVICE_BS_S1:-12}
PER_DEVICE_BS_S2=${PER_DEVICE_BS_S2:-6}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c .)}
[ "${GPUS_PER_NODE:-0}" -gt 0 ] 2>/dev/null || GPUS_PER_NODE=8
NGPU=$((NNODES * GPUS_PER_NODE))
STAGE=${STAGE:-all}
RESUME=${RESUME:-}
OFFLOAD_OPTIMIZER=${OFFLOAD_OPTIMIZER:-none}
REPORT_TO=${REPORT_TO:-none}
NPROC=$(nproc 2>/dev/null || echo 8)
NUM_PROC=${NUM_PROC:-$(( NPROC < 64 ? NPROC : 64 ))}

fatal() { echo "[prove] FATAL: $1" >&2; exit "${2:-1}"; }
[ -f "$MARS_DIR/train/train_ar_sft.py" ] && [ -f "$MARS_DIR/train/train_mars.py" ] \
  || fatal "MARS_DIR=$MARS_DIR does not contain train/train_ar_sft.py and train/train_mars.py" 2
case "$STAGE" in all|1|2) ;; *) fatal "STAGE must be all, 1 or 2 (got '$STAGE')" ;; esac
case "$OFFLOAD_OPTIMIZER" in none|cpu) ;; *) fatal "OFFLOAD_OPTIMIZER must be none or cpu" ;; esac

# gradient accumulation so that per_device x accumulation x gpus == EFF_BATCH (384 in the paper:
# 12 x 1 x 32 and 6 x 2 x 32 on 32 GPUs; 12 x 4 x 8 and 6 x 8 x 8 on one 8-GPU node)
ga_for() {
  local per=$(( $1 * NGPU ))
  if (( EFF_BATCH % per != 0 )); then
    echo "[prove] FATAL: EFF_BATCH=$EFF_BATCH is not a multiple of per_device($1) x gpus($NGPU) = $per; change PER_DEVICE_BS_S1/S2" >&2
    return 3
  fi
  echo $(( EFF_BATCH / per ))
}
GA_S1=$(ga_for "$PER_DEVICE_BS_S1") || exit 3
GA_S2=$(ga_for "$PER_DEVICE_BS_S2") || exit 3

mkdir -p "$OUT_DIR" || fatal "cannot create OUT_DIR=$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
AR_OUT=$OUT_DIR/ar_sft
MARS_OUT=$OUT_DIR/mars_blk$BLOCK_SIZE
STAGE2_INIT=${STAGE2_INIT:-$AR_OUT/checkpoint-final}
[ -d "$STAGE2_DATASET" ] && STAGE2_DATASET=$(cd "$STAGE2_DATASET" && pwd)
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/configs"
exec > >(tee -a "$OUT_DIR/logs/train_node${NODE_RANK}.log") 2>&1

# ---- runtime copies of the DeepSpeed / accelerate configs (the checked-in files are single-node) ----
# zero_hpz_partition_size = GPUs per node: with more than one node, parameters are sharded
# inside a node and replicated across nodes, so only the gradient reduce-scatter crosses the
# network (plain ZeRO-3 across 32 ranks re-gathers every layer over the wire).
CFG=$OUT_DIR/configs
HPZ=$(( NNODES > 1 ? GPUS_PER_NODE : 1 ))
sed -E -e "s/(\"zero_hpz_partition_size\": )[0-9]+/\1$HPZ/" \
       -e "s/(\"offload_optimizer\": \{ \"device\": \")[a-z]+/\1$OFFLOAD_OPTIMIZER/" \
       "$HERE/configs/ds_zero3_hpz.json" > "$CFG/ds_zero3_hpz.json"
sed -E -e "s#^(  deepspeed_config_file: ).*#\1$CFG/ds_zero3_hpz.json#" \
       -e "s/^(machine_rank: ).*/\1$NODE_RANK/" \
       -e "s/^(num_machines: ).*/\1$NNODES/" \
       -e "s/^(num_processes: ).*/\1$NGPU/" \
       "$HERE/configs/accelerate_zero3.yaml" > "$CFG/accelerate_zero3.yaml"
python -c "import json,sys; json.load(open(sys.argv[1]))" "$CFG/ds_zero3_hpz.json" || fatal "bad DeepSpeed json"

export PYTHONPATH="$MARS_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false HF_DATASETS_TRUST_REMOTE_CODE=True
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export DEEPSPEED_TIMEOUT=${DEEPSPEED_TIMEOUT:-1800} TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cat <<SUMMARY
[prove] proposer training @ $(date)
        MARS_DIR=$MARS_DIR
        base=$BASE_MODEL  stage-1 data=$DATASET  stage-2 data=${STAGE2_DATASET:-<unset>}
        out=$OUT_DIR  block=$BLOCK_SIZE  epochs=$EPOCHS  lr=$LR  max_length=$MAX_LENGTH
        nodes=$NNODES (rank $NODE_RANK) x $GPUS_PER_NODE gpus = $NGPU  hpz=$HPZ  offload=$OFFLOAD_OPTIMIZER
        effective batch $EFF_BATCH = stage1 $PER_DEVICE_BS_S1 x $GA_S1 x $NGPU, stage2 $PER_DEVICE_BS_S2 x $GA_S2 x $NGPU
        stage=$STAGE  resume=${RESUME:-none}  report_to=$REPORT_TO
SUMMARY

LAUNCH=(accelerate launch --config_file "$CFG/accelerate_zero3.yaml"
        --num_machines "$NNODES" --machine_rank "$NODE_RANK" --num_processes "$NGPU"
        --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT")
cd "$MARS_DIR" || exit 2

# ================= STAGE 1: AR SFT =================
run_stage1() {
  echo "===== STAGE 1: AR SFT  $BASE_MODEL -> $AR_OUT @ $(date) ====="
  local RES=()
  if [ -n "$RESUME" ]; then RES=(--resume_from_checkpoint "$RESUME"); RESUME=""; fi
  "${LAUNCH[@]}" train/train_ar_sft.py ${RES[@]+"${RES[@]}"} \
    --model_name_or_path "$BASE_MODEL" \
    --output_dir "$AR_OUT" \
    --dataset_args "$DATASET" \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --per_device_train_batch_size "$PER_DEVICE_BS_S1" \
    --gradient_accumulation_steps "$GA_S1" \
    --eval_strategy "no" \
    --max_length "$MAX_LENGTH" \
    --save_total_limit 2 \
    --save_steps 300 \
    --save_only_model False \
    --logging_steps 10 \
    --warmup_ratio 0.03 \
    --group_by_length \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --report_to "$REPORT_TO" \
    --dataloader_num_workers 8 --dataloader_prefetch_factor 4 --dataloader_pin_memory True \
    --bf16 \
    --num_proc "$NUM_PROC"
}

# ================= STAGE 2: block training (MARS objective, block_size 16) =================
# per-device batch is half of stage 1 because MARS concatenates [placeholder stream | original],
# doubling the tokens of every row; the effective batch stays EFF_BATCH.
run_stage2() {
  echo "===== STAGE 2: block training blk$BLOCK_SIZE  $STAGE2_INIT -> $MARS_OUT @ $(date) ====="
  local RES=()
  if [ -n "$RESUME" ]; then RES=(--resume_from_checkpoint "$RESUME"); RESUME=""; fi
  "${LAUNCH[@]}" train/train_mars.py ${RES[@]+"${RES[@]}"} \
    --model_name_or_path "$STAGE2_INIT" \
    --output_dir "$MARS_OUT" \
    --dataset_args "$STAGE2_DATASET" \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --per_device_train_batch_size "$PER_DEVICE_BS_S2" \
    --gradient_accumulation_steps "$GA_S2" \
    --attn_implementation sdpa \
    --eval_strategy "no" \
    --max_length "$MAX_LENGTH" \
    --save_total_limit 2 \
    --block_size "$BLOCK_SIZE" \
    --right_shift_logits \
    --ar_weight 1.0 \
    --save_steps 300 \
    --save_only_model False \
    --logging_steps 10 \
    --warmup_ratio 0.03 \
    --group_by_length \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --report_to "$REPORT_TO" \
    --dataloader_num_workers 8 --dataloader_prefetch_factor 4 --dataloader_pin_memory True \
    --bf16 \
    --num_proc "$NUM_PROC"
}

rc=0
if [ "$STAGE" = all ] || [ "$STAGE" = 1 ]; then
  if [ "$STAGE" = all ] && [ -d "$AR_OUT/checkpoint-final" ]; then
    echo "[prove] stage 1 already done ($AR_OUT/checkpoint-final exists), skipping"
  else
    run_stage1; rc=$?
    echo "[prove] STAGE 1 rc=$rc @ $(date)"
    [ $rc -eq 0 ] || exit $rc
    [ -d "$AR_OUT/checkpoint-final" ] || fatal "stage 1 produced no checkpoint-final" 5
  fi
fi

if [ "$STAGE" = all ] || [ "$STAGE" = 2 ]; then
  case "$STAGE2_DATASET" in
    /*|./*|../*) [ -d "$STAGE2_DATASET" ] || fatal "STAGE2_DATASET=$STAGE2_DATASET does not exist" 4 ;;
  esac
  [ -d "$STAGE2_INIT" ] || [[ "$STAGE2_INIT" == */* && ! "$STAGE2_INIT" == /* ]] \
    || fatal "stage-2 init checkpoint $STAGE2_INIT not found (run stage 1 first or set STAGE2_INIT)" 5
  run_stage2; rc=$?
  echo "[prove] STAGE 2 rc=$rc @ $(date)"
  echo "[prove] note: on the DeepSpeed path the logged 'loss' is the true loss x gradient_accumulation_steps ($GA_S2)"
  [ $rc -eq 0 ] && ls -la "$MARS_OUT/checkpoint-final" 2>/dev/null | head -3
fi

if [ $rc -eq 0 ]; then
  echo "[prove] done. Before serving, rewrite the checkpoint config for the inference stack:"
  echo "        python inference/tools/prepare_checkpoint.py $MARS_OUT/checkpoint-final <served-dir>"
fi
exit $rc
