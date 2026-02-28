#!/bin/sh
set -eu

# Learn2Reg (inter-patient) VoxelMorph runs (OASIS): train (on labeled training split) + test inference.
# Produces figures under:
#   ${CONVOLT_RESULTS_ROOT:-/scratch/yc130/Registration/outputs}/oasis_voxelmorph_{train_mode}_{atlas_tag}/figures
#
# Usage examples:
#   scripts/run_learn2reg_voxelmorph.sh unsupervised cuda
#   scripts/run_learn2reg_voxelmorph.sh supervised cuda
#   scripts/run_learn2reg_voxelmorph.sh hybrid cuda

DATASET="oasis"
TRAIN_MODE="${1:-unsupervised}"             # unsupervised|supervised|hybrid
DEVICE="${2:-cuda}"                   # cpu|cuda|auto
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

RESULTS_DIR="${OUTPUTS_ROOT}/${DATASET}_voxelmorph_${TRAIN_MODE}_${ATLAS_TAG}"
VM_TRAIN_CASES="${VM_TRAIN_CASES:-250}"
VM_EPOCHS="${VM_EPOCHS:-50}"
VM_STEPS="${VM_STEPS:-100}"
VM_SCALE="${VM_SCALE:-2}"
VM_SMOOTH="${VM_SMOOTH:-0.05}"
VM_LR="${VM_LR:-1e-4}"

# Dataset-aware defaults (can be overridden via env vars).
VM_SIM="${VM_SIM:-ncc}"
VM_NCC_WIN="${VM_NCC_WIN:-9}"

echo "== Train =="
python -m reg register \
  --dataset "${DATASET}" \
  --method voxelmorph \
  --split training \
  --results_dir "${RESULTS_DIR}" \
  --atlas_mode "${ATLAS_MODE}" \
  --atlas_n "${ATLAS_N}" \
  --atlas_seed "${ATLAS_SEED}" \
  --vm_device "${DEVICE}" \
  --vm_train_mode "${TRAIN_MODE}" \
  --vm_train_cases "${VM_TRAIN_CASES}" \
  --vm_train_epochs "${VM_EPOCHS}" \
  --vm_steps_per_epoch "${VM_STEPS}" \
  --vm_scale "${VM_SCALE}" \
  --vm_lr "${VM_LR}" \
  --vm_smooth "${VM_SMOOTH}" \
  --vm_sim "${VM_SIM}" \
  --vm_ncc_win "${VM_NCC_WIN}"

echo "== Test inference (uses saved voxelmorph.pt) =="
python -m reg register \
  --dataset "${DATASET}" \
  --method voxelmorph \
  --split test \
  --results_dir "${RESULTS_DIR}" \
  --atlas_mode "${ATLAS_MODE}" \
  --atlas_n "${ATLAS_N}" \
  --atlas_seed "${ATLAS_SEED}" \
  --vm_device "${DEVICE}" \
  --vm_train_mode none \
  --vm_train_epochs 0

echo "== Done. Check figures under ${RESULTS_DIR}/figures =="
