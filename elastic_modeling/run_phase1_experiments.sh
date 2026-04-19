#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Phase 1 smoke run"
python elastic_modeling/train_router.py \
  --steps 200 \
  --batch-size 2 \
  --max-examples 2000 \
  --save-every 100 \
  --log-every 10

echo
echo "Evaluate latest smoke checkpoint"
python elastic_modeling/eval_router.py \
  --checkpoint-path checkpoints/router_phase1/router_step_000200.pt \
  --max-examples 2000 \
  --compare-base-full-budget \
  --csv-path checkpoints/router_phase1/router_step_000200_eval.csv

echo
echo "Phase 1 short 100k run"
python elastic_modeling/train_router.py \
  --steps 2000 \
  --batch-size 2 \
  --dataset-path data/micro_fineweb_subset \
  --save-every 250 \
  --log-every 25

echo
echo "Evaluate short-run checkpoint"
python elastic_modeling/eval_router.py \
  --checkpoint-path checkpoints/router_phase1/router_step_002000.pt \
  --dataset-path data/micro_fineweb_subset \
  --compare-base-full-budget \
  --csv-path checkpoints/router_phase1/router_step_002000_eval.csv
