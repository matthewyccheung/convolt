#!/bin/sh
set -eu

# Run UQ for a single registration results folder (must contain summary.csv).
#
# Usage:
#   sh scripts/run_uq_one_output.sh ${CONVOLT_RESULTS_ROOT:-/scratch/yc130/Registration/outputs}/oasis_voxelmorph_unsupervised_atlas-multi5
#
# Optional env vars:
#   UQ_ROOT=${CONVOLT_UQ_ROOT:-uq_results}
#   ALPHA=0.1
#   N_REPEATS=50
#   BETA_MODEL=ridge|none
#   TOPK_LABELS=3
#   RUN_LUNG_REGIONS=1
#   RADIAL_SHELLS=5
#   REGION_SCORES=q90,max,mean
#   N_TRAIN_LUNGCT=10
#   N_CALIB_LUNGCT=15
#   N_TEST_LUNGCT=5

RESULTS_DIR="${1:-}"
if [ -z "${RESULTS_DIR}" ]; then
  echo "Usage: sh scripts/run_uq_one_output.sh /path/to/results_dir" >&2
  exit 2
fi
if [ ! -d "${RESULTS_DIR}" ] || [ ! -f "${RESULTS_DIR}/summary.csv" ]; then
  echo "Invalid results_dir (missing summary.csv): ${RESULTS_DIR}" >&2
  exit 2
fi

UQ_ROOT="${UQ_ROOT:-${CONVOLT_UQ_ROOT:-uq_results}}"
ALPHA="${ALPHA:-0.1}"
N_REPEATS="${N_REPEATS:-50}"
BETA_MODEL="${BETA_MODEL:-ridge}"
TOPK_LABELS="${TOPK_LABELS:-3}"
RUN_LOCAL_SCP="${RUN_LOCAL_SCP:-1}"
SCP_LOCAL_S="${SCP_LOCAL_S:-abs_pred}"
SCP_LOCAL_K="${SCP_LOCAL_K:-50}"
RUN_LUNG_REGIONS="${RUN_LUNG_REGIONS:-1}"
RADIAL_SHELLS="${RADIAL_SHELLS:-5}"
REGION_SCORES="${REGION_SCORES:-q90,max,mean}"
N_TRAIN_LUNGCT="${N_TRAIN_LUNGCT:-10}"
N_CALIB_LUNGCT="${N_CALIB_LUNGCT:-15}"
N_TEST_LUNGCT="${N_TEST_LUNGCT:-5}"

name="$(basename "${RESULTS_DIR}")"
dataset="$(printf "%s" "${name}" | awk -F_ '{print $1}')"

detect_method() {
  n="$1"
  for m in itk_elastix_bspline sitk_diffeomorphic_demons sitk_bspline voxelmorph demons; do
    case "${n}" in
      *_${m}|*_${m}_*) echo "${m}"; return 0 ;;
    esac
  done
  echo ""
}

method="$(detect_method "${name}")"
if [ -z "${method}" ]; then
  echo "Could not detect method from folder name: ${name}" >&2
  exit 2
fi

out_dir="${UQ_ROOT}/${name}"
mkdir -p "${out_dir}"

learn2reg_remaining() {
  python - "$1" <<'PY'
import csv
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
summary = results_dir / "summary.csv"
meta = results_dir / "atlas_meta.json"

ids = set()
with summary.open(newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        pid = row.get("patient_id", "")
        if pid:
            ids.add(str(pid))

exclude = set()
if meta.exists():
    try:
        m = json.loads(meta.read_text())
        for k in ("atlas_ids", "vm_train_ids"):
            for x in m.get(k, []) or []:
                exclude.add(str(x))
    except Exception:
        pass

print(len(ids - exclude))
PY
}

learn2reg_total() {
  python - "$1" <<'PY'
import csv
import sys
from pathlib import Path

summary = Path(sys.argv[1]) / "summary.csv"
ids = set()
with summary.open(newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        pid = row.get("patient_id", "")
        if pid:
            ids.add(str(pid))
print(len(ids))
PY
}

extra=""
if [ "${RUN_LOCAL_SCP}" = "1" ]; then
  extra="${extra} --scp_local --scp_local_s ${SCP_LOCAL_S} --scp_knn_k ${SCP_LOCAL_K}"
fi
case "${dataset}" in
  oasis)
    remain="$(learn2reg_remaining "${RESULTS_DIR}")"
    total="$(learn2reg_total "${RESULTS_DIR}")"
    required_min=10
    if [ "${remain}" -lt "${required_min}" ]; then
      echo "SKIP (not enough evaluable labeled cases after excluding atlas/vm_train ids): ${name}" >&2
      echo "total_labeled=${total} remaining_for_uq=${remain} required_min=${required_min}" >&2
      echo "Hint: rerun registration with a smaller --vm_train_cases and/or --atlas_n to leave held-out subjects for UQ." >&2
      exit 0
    fi
    extra="${extra} --uq_target volume_union --uq_topk_labels ${TOPK_LABELS}"
    ;;
  nlst)
    if [ "${RUN_LUNG_REGIONS}" = "1" ]; then
      extra="${extra} --region_defs radial --radial_shells ${RADIAL_SHELLS} --region_scores ${REGION_SCORES}"
    fi
    ;;
  lungct)
    if [ "${RUN_LUNG_REGIONS}" = "1" ]; then
      extra="${extra} --region_defs radial --radial_shells ${RADIAL_SHELLS} --region_scores ${REGION_SCORES}"
    fi
    extra="${extra} --n_train ${N_TRAIN_LUNGCT} --n_calib ${N_CALIB_LUNGCT} --n_test ${N_TEST_LUNGCT}"
    ;;
  acdc)
    ;;
  *)
    echo "Unknown dataset=${dataset} (parsed from ${name}). Override by running python -m reg.uq.cli manually." >&2
    exit 2
    ;;
esac

echo "UQ: results_dir=${RESULTS_DIR}"
echo "UQ: out_dir=${out_dir}"
echo "UQ: dataset=${dataset} method=${method} alpha=${ALPHA} n_repeats=${N_REPEATS} beta_model=${BETA_MODEL}"

# shellcheck disable=SC2086
if python -m reg.uq.cli \
    --dataset "${dataset}" \
    --method "${method}" \
    --results_dir "${RESULTS_DIR}" \
    --out_dir "${out_dir}" \
    --alpha "${ALPHA}" \
    --n_repeats "${N_REPEATS}" \
    --beta_model "${BETA_MODEL}" \
    --uq_topk_labels "${TOPK_LABELS}" \
    ${extra}; then
  :
else
  code="$?"
  echo "FAIL (exit=${code}): ${name}" >&2
  exit "${code}"
fi
