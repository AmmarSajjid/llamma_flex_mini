#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_SMOKE_RUN="${RUN_SMOKE_RUN:-1}"
SMOKE_STEPS="${SMOKE_STEPS:-200}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
SMOKE_GRAD_ACCUM="${SMOKE_GRAD_ACCUM:-4}"
SMOKE_MAX_EXAMPLES="${SMOKE_MAX_EXAMPLES:-2000}"

SHORT_STEPS="${SHORT_STEPS:-12500}"
SHORT_BATCH_SIZE="${SHORT_BATCH_SIZE:-8}"
SHORT_GRAD_ACCUM="${SHORT_GRAD_ACCUM:-2}"
SHORT_EVAL_MAX_EXAMPLES="${SHORT_EVAL_MAX_EXAMPLES:-5000}"

SHORT_STEP_TAG="$(printf "%06d" "$SHORT_STEPS")"

if [[ "$RUN_SMOKE_RUN" == "1" ]]; then
  echo "Phase 1 smoke run"
  python elastic_modeling/train_router.py \
    --steps "$SMOKE_STEPS" \
    --batch-size "$SMOKE_BATCH_SIZE" \
    --grad-accum-steps "$SMOKE_GRAD_ACCUM" \
    --max-examples "$SMOKE_MAX_EXAMPLES" \
    --warmup-ratio 0.10 \
    --fail-on-nan \
    --save-failure-state \
    --skip-non-finite-steps \
    --save-every 100 \
    --log-every 10

  echo
  echo "Evaluate latest smoke checkpoint"
  python elastic_modeling/eval_router.py \
    --checkpoint-path "checkpoints/router_phase1/router_step_$(printf "%06d" "$SMOKE_STEPS").pt" \
    --max-examples "$SMOKE_MAX_EXAMPLES" \
    --compare-base-full-budget \
    --csv-path "checkpoints/router_phase1/router_step_$(printf "%06d" "$SMOKE_STEPS")_eval.csv"
fi

echo
echo "Phase 1 short 100k run"
python elastic_modeling/train_router.py \
  --steps "$SHORT_STEPS" \
  --batch-size "$SHORT_BATCH_SIZE" \
  --grad-accum-steps "$SHORT_GRAD_ACCUM" \
  --dataset-path data/micro_fineweb_subset \
  --warmup-ratio 0.10 \
  --fail-on-nan \
  --save-failure-state \
  --skip-non-finite-steps \
  --save-every 250 \
  --log-every 25

echo
echo "Evaluate short-run checkpoint"
python elastic_modeling/eval_router.py \
  --checkpoint-path "checkpoints/router_phase1/router_step_${SHORT_STEP_TAG}.pt" \
  --dataset-path data/micro_fineweb_subset \
  --max-examples "$SHORT_EVAL_MAX_EXAMPLES" \
  --compare-base-full-budget \
  --csv-path "checkpoints/router_phase1/router_step_${SHORT_STEP_TAG}_eval.csv"
