#!/bin/sh
set -eu

# Run UQ for every registration results folder under ${CONVOLT_RESULTS_ROOT:-/scratch/yc130/Registration/outputs}
# that contains a summary.csv. Writes to uq_results/{folder_name} by default.
#
# Environment variables (optional):
#   OUTPUTS_ROOT=${CONVOLT_RESULTS_ROOT:-/scratch/yc130/Registration/outputs}
#   UQ_ROOT=uq_results
#   ALPHA=0.1
#   N_REPEATS=50
#   BETA_MODEL=ridge|none
#   TOPK_LABELS=0|K|all
#     - 0/none: union volume only (fast; default)
#     - K: run Learn2Reg "volume suite" (union + top-K labels by mean GT volume)
#     - all: run Learn2Reg "volume suite" (union + all labels in label_volumes.csv, excluding label_id=0)
#   RUN_LUNG_REGIONS=1
#   RADIAL_SHELLS=5
#   REGION_SCORES=q90,max,mean
#   REGION_BETA_MODES=region|global|both
#   N_TRAIN_LUNGCT=10
#   N_CALIB_LUNGCT=15
#   N_TEST_LUNGCT=5
#   N_TRAIN_ABDOMENCTCT= (optional override; requires enough held-out cases)
#   N_CALIB_ABDOMENCTCT=
#   N_TEST_ABDOMENCTCT=
#   ONLY_DATASET=...   (e.g. oasis)
#   ONLY_METHOD=...    (e.g. voxelmorph)
#
# Example:
#   ALPHA=0.1 N_REPEATS=100 sh scripts/run_uq_all_outputs.sh

OUTPUTS_ROOT="${OUTPUTS_ROOT:-${CONVOLT_RESULTS_ROOT:-/scratch/yc130/Registration/outputs}}"
UQ_ROOT="${UQ_ROOT:-${CONVOLT_UQ_ROOT:-uq_results}}"
ALPHA="${ALPHA:-0.1}"
N_REPEATS="${N_REPEATS:-100}"
BETA_MODEL="${BETA_MODEL:-ridge}"
TOPK_LABELS="${TOPK_LABELS:-all}"
RUN_LOCAL_SCP="${RUN_LOCAL_SCP:-1}"
SCP_LOCAL_S="${SCP_LOCAL_S:-abs_pred}"
SCP_LOCAL_K="${SCP_LOCAL_K:-50}"
RUN_LUNG_REGIONS="${RUN_LUNG_REGIONS:-1}"
RADIAL_SHELLS="${RADIAL_SHELLS:-5}"
REGION_SCORES="${REGION_SCORES:-q90,max,mean}"
REGION_BETA_MODES="${REGION_BETA_MODES:-both}"
N_TRAIN_LUNGCT="${N_TRAIN_LUNGCT:-10}"
N_CALIB_LUNGCT="${N_CALIB_LUNGCT:-15}"
N_TEST_LUNGCT="${N_TEST_LUNGCT:-5}"
N_TRAIN_ABDOMENCTCT="${N_TRAIN_ABDOMENCTCT:-}"
N_CALIB_ABDOMENCTCT="${N_CALIB_ABDOMENCTCT:-}"
N_TEST_ABDOMENCTCT="${N_TEST_ABDOMENCTCT:-}"
ONLY_DATASET="${ONLY_DATASET:-}"
ONLY_METHOD="${ONLY_METHOD:-}"

if [ ! -d "${OUTPUTS_ROOT}" ]; then
  echo "Missing OUTPUTS_ROOT=${OUTPUTS_ROOT}" >&2
  exit 2
fi

mkdir -p "${UQ_ROOT}"

detect_method() {
  # Input: folder basename (dataset_method_...); output: method string or empty.
  name="$1"
  # Known methods (longest first).
  for m in itk_elastix_bspline sitk_diffeomorphic_demons sitk_bspline voxelmorph demons; do
    case "${name}" in
      *_${m}|*_${m}_*) echo "${m}"; return 0 ;;
    esac
  done
  echo ""
}

learn2reg_remaining() {
  # Print remaining labeled case count after excluding atlas_ids/vm_train_ids from atlas_meta.json (if present).
  # Usage: learn2reg_remaining /path/to/results_dir
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

remain = sorted(ids - exclude)
print(len(remain))
PY
}

learn2reg_total() {
  # Print total labeled case count in summary.csv (before exclusions).
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

learn2reg_label_list() {
  # Print all non-background label IDs from label_volumes.csv as a comma-separated list.
  # Usage: learn2reg_label_list /path/to/results_dir
  python - "$1" <<'PY'
import sys
from pathlib import Path

import pandas as pd

results_dir = Path(sys.argv[1])
lv = results_dir / "label_volumes.csv"
if not lv.exists():
    print("")
    raise SystemExit(0)

df = pd.read_csv(lv, usecols=["label_id"])
labs = sorted({int(x) for x in df["label_id"].dropna().tolist() if int(x) > 0})
print(",".join(map(str, labs)))
PY
}

echo "UQ sweep: outputs_root=${OUTPUTS_ROOT} uq_root=${UQ_ROOT} alpha=${ALPHA} n_repeats=${N_REPEATS}"

for d in "${OUTPUTS_ROOT}"/*; do
  [ -d "$d" ] || continue
  [ -f "$d/summary.csv" ] || continue

  name="$(basename "$d")"
  dataset="$(printf "%s" "$name" | awk -F_ '{print $1}')"
  method="$(detect_method "$name")"
  if [ -z "${method}" ]; then
    echo "SKIP (unknown method): ${name}"
    continue
  fi

  if [ -n "${ONLY_DATASET}" ] && [ "${dataset}" != "${ONLY_DATASET}" ]; then
    continue
  fi
  if [ -n "${ONLY_METHOD}" ] && [ "${method}" != "${ONLY_METHOD}" ]; then
    continue
  fi

  out_dir="${UQ_ROOT}/${name}"

  # Decide dataset-specific flags.
  extra=""
  if [ "${RUN_LOCAL_SCP}" = "1" ]; then
    extra="${extra} --scp_local --scp_local_s ${SCP_LOCAL_S} --scp_knn_k ${SCP_LOCAL_K}"
  fi
  case "${dataset}" in
    hippocampusmr|oasis|abdomenctct)
      # If atlas/vm_train excluded all labeled cases, UQ cannot run.
      remain="$(learn2reg_remaining "$d")"
      total="$(learn2reg_total "$d")"
      # If user configured explicit split sizes (abdomenctct), ensure we have enough held-out cases.
      required_min=10
      use_abd_override=0
      if [ "${dataset}" = "abdomenctct" ] && [ -n "${N_TRAIN_ABDOMENCTCT}" ] && [ -n "${N_CALIB_ABDOMENCTCT}" ] && [ -n "${N_TEST_ABDOMENCTCT}" ]; then
        nt="${N_TRAIN_ABDOMENCTCT}"
        nc="${N_CALIB_ABDOMENCTCT}"
        nte="${N_TEST_ABDOMENCTCT}"
        required_min=$((nt + nc + nte))
        use_abd_override=1
      fi
      if [ "${remain}" -lt "${required_min}" ]; then
        echo "SKIP (not enough evaluable labeled cases after excluding atlas/vm_train ids): ${name}"
        echo "  total_labeled=${total} remaining_for_uq=${remain} required_min=${required_min}"
        echo "  Hint: rerun registration with a smaller --vm_train_cases and/or --atlas_n to leave held-out subjects for UQ."
        continue
      fi

      # Learn2Reg targets:
      # - default/fast: union volume only
      # - if TOPK_LABELS is K>0: union + top-K labels
      # - if TOPK_LABELS=all: union + all labels (label_id>0)
      case "${TOPK_LABELS}" in
        ""|0|none)
          extra="${extra} --uq_target volume_union"
          ;;
        all)
          label_list="$(learn2reg_label_list "$d")"
          if [ -z "${label_list}" ]; then
            echo "SKIP (missing label_volumes.csv for all-label UQ): ${name}"
            echo "  Hint: rerun registration on split=training to generate label_volumes.csv."
            continue
          fi
          # Omit --uq_target so reg.uq.cli runs the Learn2Reg volume suite: union + labels.
          extra="${extra} --uq_label_list ${label_list}"
          ;;
        *)
          # Omit --uq_target so reg.uq.cli runs the Learn2Reg volume suite: union + top-K labels.
          extra="${extra} --uq_topk_labels ${TOPK_LABELS}"
          ;;
      esac

      if [ "${use_abd_override}" = "1" ]; then
        extra="${extra} --n_train ${N_TRAIN_ABDOMENCTCT} --n_calib ${N_CALIB_ABDOMENCTCT} --n_test ${N_TEST_ABDOMENCTCT}"
      fi
      ;;
    nlst)
      # global + (optional) regional (radial) in one run
      if [ "${RUN_LUNG_REGIONS}" = "1" ]; then
        extra="${extra} --region_defs radial --radial_shells ${RADIAL_SHELLS} --region_scores ${REGION_SCORES}"
      fi
      ;;
    lungct)
      if [ "${RUN_LUNG_REGIONS}" = "1" ]; then
        extra="${extra} --region_defs radial --radial_shells ${RADIAL_SHELLS} --region_scores ${REGION_SCORES}"
      fi
      # For small LungCT (n=30), use fixed split sizes.
      extra="${extra} --n_train ${N_TRAIN_LUNGCT} --n_calib ${N_CALIB_LUNGCT} --n_test ${N_TEST_LUNGCT}"
      ;;
    acdc)
      # leave defaults (delta volume / lvef pipeline as implemented)
      ;;
    *)
      echo "SKIP (unknown dataset): ${name}"
      continue
      ;;
  esac

  run_one() {
    mode="$1"
    suffix="$2"
    out_dir_mode="${out_dir}${suffix}"
    extra_mode="${extra}"
    if [ "${RUN_LUNG_REGIONS}" = "1" ] && [ -n "${mode}" ]; then
      extra_mode="${extra_mode} --region_beta_mode ${mode}"
    fi

    echo "RUN: ${name}${suffix}"
    # Note: pass results_dir/out_dir explicitly so this works even when results folders include train_mode/atlas tags.
    # If a particular folder is not runnable (e.g., insufficient data after exclusions), report and continue.
    # shellcheck disable=SC2086
    if python -m reg.uq.cli \
        --dataset "${dataset}" \
        --method "${method}" \
        --results_dir "${d}" \
        --out_dir "${out_dir_mode}" \
        --alpha "${ALPHA}" \
        --n_repeats "${N_REPEATS}" \
        --beta_model "${BETA_MODEL}" \
        ${extra_mode}; then
      :
    else
      code="$?"
      echo "FAIL (exit=${code}): ${name}${suffix}" >&2
    fi
  }

  case "${REGION_BETA_MODES}" in
    both)
      # Produce both locality variants for region guarantees.
      run_one "region" ""
      run_one "global" "_globalfeat"
      ;;
    global)
      run_one "global" ""
      ;;
    region|*)
      run_one "region" ""
      ;;
  esac
done

echo "Done. UQ results under ${UQ_ROOT}/"
