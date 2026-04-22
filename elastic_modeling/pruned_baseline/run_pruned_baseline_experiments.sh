#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_SMOKE_RUN="${RUN_SMOKE_RUN:-1}"
BASELINE_BUDGETS="${BASELINE_BUDGETS:-0.25 0.5 0.75 1.0}"
WIDTH_GRANULARITY="${WIDTH_GRANULARITY:-1}"

SMOKE_STEPS="${SMOKE_STEPS:-200}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
SMOKE_GRAD_ACCUM="${SMOKE_GRAD_ACCUM:-4}"
SMOKE_MAX_EXAMPLES="${SMOKE_MAX_EXAMPLES:-2000}"

SHORT_STEPS="${SHORT_STEPS:-12500}"
SHORT_BATCH_SIZE="${SHORT_BATCH_SIZE:-8}"
SHORT_GRAD_ACCUM="${SHORT_GRAD_ACCUM:-2}"
SHORT_EVAL_MAX_EXAMPLES="${SHORT_EVAL_MAX_EXAMPLES:-5000}"

run_one_budget() {
  local budget="$1"
  local steps="$2"
  local batch_size="$3"
  local grad_accum="$4"
  local train_max_examples="$5"
  local eval_max_examples="$6"

  local budget_tag="${budget//./p}"
  local step_tag
  step_tag="$(printf "%06d" "$steps")"
  local save_dir="checkpoints/pruned_baseline/budget_${budget_tag}"

  python elastic_modeling/pruned_baseline/train_pruned_baseline.py \
    --target-budget "$budget" \
    --width-granularity "$WIDTH_GRANULARITY" \
    --save-dir "$save_dir" \
    --steps "$steps" \
    --batch-size "$batch_size" \
    --grad-accum-steps "$grad_accum" \
    --max-examples "$train_max_examples" \
    --warmup-ratio 0.10 \
    --fail-on-nan \
    --save-failure-state \
    --skip-non-finite-steps \
    --save-every 250 \
    --log-every 25

  python elastic_modeling/pruned_baseline/eval_pruned_baseline.py \
    --checkpoint-path "${save_dir}/pruned_step_${step_tag}.pt" \
    --dataset-path data/micro_fineweb_subset \
    --max-examples "$eval_max_examples" \
    --compare-base-full-budget \
    --csv-path "${save_dir}/pruned_step_${step_tag}_eval.csv"
}

if [[ "$RUN_SMOKE_RUN" == "1" ]]; then
  echo "Pruned baseline smoke runs"
  for budget in $BASELINE_BUDGETS; do
    echo
    echo "Smoke budget ${budget}"
    run_one_budget \
      "$budget" \
      "$SMOKE_STEPS" \
      "$SMOKE_BATCH_SIZE" \
      "$SMOKE_GRAD_ACCUM" \
      "$SMOKE_MAX_EXAMPLES" \
      "$SMOKE_MAX_EXAMPLES"
  done
fi

echo
echo "Pruned baseline short 100k runs"
for budget in $BASELINE_BUDGETS; do
  echo
  echo "Short-run budget ${budget}"
  run_one_budget \
    "$budget" \
    "$SHORT_STEPS" \
    "$SHORT_BATCH_SIZE" \
    "$SHORT_GRAD_ACCUM" \
    0 \
    "$SHORT_EVAL_MAX_EXAMPLES"
done
