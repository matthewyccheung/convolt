#!/bin/sh
set -eu

# Learn2Reg (inter-patient) Demons runs (OASIS): atlas-based segmentation via classical registration.
#
# Produces outputs under:
#   ${CONVOLT_RESULTS_ROOT:-/scratch/yc130/Registration/outputs}/oasis_demons_{atlas_tag}
#
# Usage:
#   sh scripts/run_learn2reg_demons.sh
#
# Optional env vars:
#   ATLAS_MODE=multi|single|average   (default: multi)
#   ATLAS_N=5                        (default: 5, for multi/average)
#   ATLAS_SEED=0                     (default: 0)

DATASET="oasis"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${CONVOLT_RESULTS_ROOT:-/scratch/yc130/Registration/outputs}}"

ATLAS_MODE="${ATLAS_MODE:-multi}"
ATLAS_N="${ATLAS_N:-5}"
ATLAS_SEED="${ATLAS_SEED:-0}"

case "${ATLAS_MODE}" in
  multi) ATLAS_TAG="atlas-multi${ATLAS_N}" ;;
  single) ATLAS_TAG="atlas-single" ;;
  average) ATLAS_TAG="atlas-avg${ATLAS_N}" ;;
  *)
    echo "Unknown ATLAS_MODE=${ATLAS_MODE} (expected: multi|single|average)" >&2
    exit 2
    ;;
esac

RESULTS_DIR="${OUTPUTS_ROOT}/${DATASET}_demons_${ATLAS_TAG}"

echo "== Train (labeled targets; excludes atlas subjects) =="
python -m reg register \
  --dataset "${DATASET}" \
  --method demons \
  --split training \
  --results_dir "${RESULTS_DIR}" \
  --atlas_mode "${ATLAS_MODE}" \
  --atlas_n "${ATLAS_N}" \
  --atlas_seed "${ATLAS_SEED}"

echo "== Test inference (unlabeled targets, if present) =="
python -m reg register \
  --dataset "${DATASET}" \
  --method demons \
  --split test \
  --results_dir "${RESULTS_DIR}" \
  --atlas_mode "${ATLAS_MODE}" \
  --atlas_n "${ATLAS_N}" \
  --atlas_seed "${ATLAS_SEED}"

echo "== Done. Check figures under ${RESULTS_DIR}/figures =="
