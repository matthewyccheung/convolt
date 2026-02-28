#!/bin/sh
set -eu

# One-stop registration runner for:
#   - Intra-patient: nlst, lungct, acdc
#   - Learn2Reg inter-patient (atlas-based segmentation via registration): oasis
#
# Runs Demons + VoxelMorph for each dataset and writes to standardized outputs under:
#   ${CONVOLT_RESULTS_ROOT:-/scratch/yc130/Registration/outputs}/...
#
# Usage:
#   sh scripts/run_registration_all.sh
#   ONLY_DATASET=lungct sh scripts/run_registration_all.sh
#   ONLY_DATASET=oasis TRAIN_MODE=supervised DEVICE=cuda sh scripts/run_registration_all.sh
#
# Optional env vars:
#   ONLY_DATASET=...             (run just one dataset)
#   ONLY_METHOD=demons|voxelmorph
#   DEVICE=cpu|cuda|auto         (default: cuda)
#
# Learn2Reg atlas options:
#   ATLAS_MODE=multi|single|average  (default: multi)
#   ATLAS_N=5                        (default: 5)
#   ATLAS_SEED=0                     (default: 0)
#
# Learn2Reg VoxelMorph options:
#   TRAIN_MODE=unsupervised|supervised|hybrid  (default: hybrid)
#   VM_TRAIN_CASES=...                          (default is dataset-aware inside the VM scripts; here default 200)
#
# Intra-patient VoxelMorph options:
#   VM_EPOCHS=50
#   VM_STEPS=100
#   VM_LR=1e-4
#   VM_SMOOTH=0.05
#   VM_SCALE=1

ONLY_DATASET="${ONLY_DATASET:-}"
ONLY_METHOD="${ONLY_METHOD:-}"
DEVICE="${DEVICE:-cuda}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${CONVOLT_RESULTS_ROOT:-/scratch/yc130/Registration/outputs}}"

ATLAS_MODE="${ATLAS_MODE:-multi}"
ATLAS_N="${ATLAS_N:-5}"
ATLAS_SEED="${ATLAS_SEED:-0}"

TRAIN_MODE="${TRAIN_MODE:-supervised}"
VM_TRAIN_CASES="${VM_TRAIN_CASES:-200}"

VM_EPOCHS="${VM_EPOCHS:-50}"
VM_STEPS="${VM_STEPS:-100}"
VM_LR="${VM_LR:-1e-4}"
VM_SMOOTH="${VM_SMOOTH:-0.05}"
VM_SCALE="${VM_SCALE:-1}"
VM_SIM="${VM_SIM:-ncc}"
VM_NCC_WIN="${VM_NCC_WIN:-9}"

atlas_tag() {
  case "${ATLAS_MODE}" in
    multi) echo "atlas-multi${ATLAS_N}" ;;
    single) echo "atlas-single" ;;
    average) echo "atlas-avg${ATLAS_N}" ;;
    *)
      echo "Unknown ATLAS_MODE=${ATLAS_MODE} (expected: multi|single|average)" >&2
      exit 2
      ;;
  esac
}

run_demons() {
  dataset="$1"
  split="$2"
  echo "== ${dataset} demons (${split}) =="
  if [ "${dataset}" = "oasis" ]; then
    tag="$(atlas_tag)"
    results_dir="${OUTPUTS_ROOT}/${dataset}_demons_${tag}"
    python -m reg register \
      --dataset "${dataset}" \
      --method demons \
      --split "${split}" \
      --results_dir "${results_dir}" \
      --atlas_mode "${ATLAS_MODE}" \
      --atlas_n "${ATLAS_N}" \
      --atlas_seed "${ATLAS_SEED}"
  else
    python -m reg register \
      --dataset "${dataset}" \
      --method demons \
      --split "${split}"
  fi
}

run_voxelmorph() {
  dataset="$1"
  split="$2"
  echo "== ${dataset} voxelmorph (${split}) =="
  if [ "${dataset}" = "oasis" ]; then
    tag="$(atlas_tag)"
    results_dir="${OUTPUTS_ROOT}/${dataset}_voxelmorph_${TRAIN_MODE}_${tag}"
    if [ "${split}" = "training" ]; then
      python -m reg register \
        --dataset "${dataset}" \
        --method voxelmorph \
        --split training \
        --results_dir "${results_dir}" \
        --atlas_mode "${ATLAS_MODE}" \
        --atlas_n "${ATLAS_N}" \
        --atlas_seed "${ATLAS_SEED}" \
        --vm_device "${DEVICE}" \
        --vm_train_mode "${TRAIN_MODE}" \
        --vm_train_cases "${VM_TRAIN_CASES}" \
        --vm_train_epochs "${VM_EPOCHS}" \
        --vm_steps_per_epoch "${VM_STEPS}" \
        --vm_scale 2 \
        --vm_lr "${VM_LR}" \
        --vm_smooth "${VM_SMOOTH}" \
        --vm_sim "${VM_SIM}" \
        --vm_ncc_win "${VM_NCC_WIN}"
    else
      # Inference only (uses saved results_dir/voxelmorph.pt).
      python -m reg register \
        --dataset "${dataset}" \
        --method voxelmorph \
        --split test \
        --results_dir "${results_dir}" \
        --atlas_mode "${ATLAS_MODE}" \
        --atlas_n "${ATLAS_N}" \
        --atlas_seed "${ATLAS_SEED}" \
        --vm_device "${DEVICE}" \
        --vm_train_mode none \
        --vm_train_epochs 0
    fi
  else
    # Intra-patient.
    extra=""
    if [ "${dataset}" = "acdc" ]; then
      extra="--vm_sim ncc --vm_ncc_win ${VM_NCC_WIN}"
    fi
    # shellcheck disable=SC2086
    python -m reg register \
      --dataset "${dataset}" \
      --method voxelmorph \
      --split "${split}" \
      --vm_device "${DEVICE}" \
      --vm_train_epochs "${VM_EPOCHS}" \
      --vm_steps_per_epoch "${VM_STEPS}" \
      --vm_scale "${VM_SCALE}" \
      --vm_lr "${VM_LR}" \
      --vm_smooth "${VM_SMOOTH}" \
      ${extra}
  fi
}

run_dataset() {
  dataset="$1"

  # Choose split(s) per dataset family.
  if [ "${dataset}" = "lungct" ]; then
    splits="all"
  elif [ "${dataset}" = "acdc" ]; then
    splits="training"
  elif [ "${dataset}" = "nlst" ]; then
    splits="training"
  else
    # Learn2Reg
    splits="training test"
  fi

  for split in ${splits}; do
    if [ -z "${ONLY_METHOD}" ] || [ "${ONLY_METHOD}" = "demons" ]; then
      run_demons "${dataset}" "${split}"
    fi
    if [ -z "${ONLY_METHOD}" ] || [ "${ONLY_METHOD}" = "voxelmorph" ]; then
      run_voxelmorph "${dataset}" "${split}"
    fi
  done
}

DATASETS="nlst lungct acdc oasis"
for d in ${DATASETS}; do
  if [ -n "${ONLY_DATASET}" ] && [ "${d}" != "${ONLY_DATASET}" ]; then
    continue
  fi
  run_dataset "${d}"
done

echo "== Done. Outputs under ${OUTPUTS_ROOT} =="
