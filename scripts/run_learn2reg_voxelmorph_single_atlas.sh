#!/bin/sh
set -eu

# Learn2Reg (inter-patient) VoxelMorph runs: train (on labeled training split) + test inference,
# using SINGLE-ATLAS transfer (atlas_mode=single) to avoid fusion effects while debugging.
#
# Produces figures under:
#   /scratch/yc130/Registration/outputs/{dataset}_voxelmorph_atlas-single/figures
#
# Usage examples:
#   scripts/run_learn2reg_voxelmorph_single_atlas.sh hippocampusmr supervised cuda
#   scripts/run_learn2reg_voxelmorph_single_atlas.sh oasis hybrid cuda
#   scripts/run_learn2reg_voxelmorph_single_atlas.sh abdomenctct unsupervised cuda

DATASET="${1:-hippocampusmr}"          # hippocampusmr|oasis|abdomenctct
TRAIN_MODE="${2:-unsupervised}"             # unsupervised|supervised|hybrid
DEVICE="${3:-cuda}"                   # cpu|cuda|auto

ATLAS_MODE="single"
ATLAS_N=1
ATLAS_SEED="${ATLAS_SEED:-0}"

ATLAS_TAG="atlas-single"
RESULTS_DIR="/scratch/yc130/Registration/outputs/${DATASET}_voxelmorph_${TRAIN_MODE}_${ATLAS_TAG}"

default_vm_train_cases() {
  case "$1" in
    abdomenctct) echo 10 ;;
    hippocampusmr) echo 150 ;;
    oasis) echo 250 ;;
    *) echo 200 ;;
  esac
}

VM_TRAIN_CASES="${VM_TRAIN_CASES:-$(default_vm_train_cases "${DATASET}")}"
VM_EPOCHS="${VM_EPOCHS:-200}"
VM_STEPS="${VM_STEPS:-100}"
VM_SCALE="${VM_SCALE:-2}"
VM_SMOOTH="${VM_SMOOTH:-0.05}"
VM_LR="${VM_LR:-1e-4}"

# Dataset-aware defaults (can be overridden via env vars).
VM_SIM="${VM_SIM:-ncc}"
VM_NCC_WIN="${VM_NCC_WIN:-9}"
if [ "${DATASET}" = "abdomenctct" ]; then
  VM_SIM="${VM_SIM:-mse}"
  VM_SCALE="${VM_SCALE:-2}"
fi

echo "== Train (atlas_mode=single) =="
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
