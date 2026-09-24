#!/usr/bin/env bash
# MemoryAgentBench 200Q full sweep launcher.
#
# Intended usage:
#   bash scripts/run_memoryagentbench_full_sweep.sh
#
# The script is restartable: each method writes a stable answers JSONL under
# its cell directory, and examples/run_benchmark.py skips existing rows.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

PIXI_PY="${PIXI_PY:-${REPO_ROOT}/.pixi/envs/default/bin/python}"
if [[ ! -x "${PIXI_PY}" ]]; then
  PIXI_PY="$(command -v python)"
fi

RUN_TAG="${RUN_TAG:-mab200_qwen32_$(date -u +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/memoryagentbench/${RUN_TAG}}"
LOG_DIR="${RESULTS_ROOT}/logs"
DATASET="${DATASET:-datasets/memoryagentbench/mab_200q.jsonl}"
MANIFEST="${MANIFEST:-datasets/memoryagentbench/mab_200q_manifest.json}"

LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-32B}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}"
LLM_BASE_URL="${LLM_BASE_URL:-${OPENAI_BASE_URL:-http://localhost:30000/v1}}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-${EMBED_BASE_URL:-http://localhost:8001/v1}}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-${OPENAI_API_KEY}}"

# Method sets:
#   tier0: smoke all methods on 24 questions.
#   default7: cheaper/default Table 1 methods.
#   heavy4: heavier Table 1 methods.
TIER="${TIER:-tier0}"
DEFAULT7_METHODS="${DEFAULT7_METHODS:-longcontext simplemem lightmem hipporag}"
HEAVY4_METHODS="${HEAVY4_METHODS:-plugmem memt ama_agent memrl dci_lite dci_lite_sum automem}"
FULL_METHODS="${FULL_METHODS:-${DEFAULT7_METHODS} ${HEAVY4_METHODS}}"
METHODS="${METHODS:-}"

case "${TIER}" in
  tier0)
    METHODS="${METHODS:-${FULL_METHODS}}"
    MAX_QUESTIONS="${MAX_QUESTIONS:-24}"
    ;;
  default7)
    METHODS="${METHODS:-${DEFAULT7_METHODS}}"
    MAX_QUESTIONS="${MAX_QUESTIONS:-200}"
    ;;
  heavy4)
    METHODS="${METHODS:-${HEAVY4_METHODS}}"
    MAX_QUESTIONS="${MAX_QUESTIONS:-200}"
    ;;
  full)
    METHODS="${METHODS:-${FULL_METHODS}}"
    MAX_QUESTIONS="${MAX_QUESTIONS:-200}"
    ;;
  *)
    echo "Unknown TIER=${TIER}; use tier0, default7, heavy4, or full." >&2
    exit 2
    ;;
esac

mkdir -p "${RESULTS_ROOT}" "${LOG_DIR}" datasets/memoryagentbench

echo "[mab] run_tag=${RUN_TAG}"
echo "[mab] tier=${TIER}"
echo "[mab] methods=${METHODS}"
echo "[mab] results=${RESULTS_ROOT}"
echo "[mab] llm=${LLM_MODEL} @ ${LLM_BASE_URL}"
echo "[mab] embed=${EMBED_MODEL} @ ${EMBEDDING_BASE_URL}"

if [[ ! -f "${DATASET}" || ! -f "${MANIFEST}" ]]; then
  echo "[mab] dataset/manifest missing; building deterministic 200Q subset"
  "${PIXI_PY}" scripts/select_memoryagentbench_200q.py \
    --output "${DATASET}" \
    --manifest "${MANIFEST}" \
    --summary-md datasets/memoryagentbench/mab_200q_summary.md
fi

[[ -s "${DATASET}" ]] || { echo "[mab] dataset file is missing or empty: ${DATASET}" >&2; exit 5; }
[[ -s "${MANIFEST}" ]] || { echo "[mab] manifest file is missing or empty: ${MANIFEST}" >&2; exit 6; }

echo "[mab] verifying endpoints"
LLM_MODELS_TMP="$(mktemp -t mab_llm_models.XXXXXX)"
EMBED_MODELS_TMP="$(mktemp -t mab_embed_models.XXXXXX)"
trap 'rm -f "${LLM_MODELS_TMP:-}" "${EMBED_MODELS_TMP:-}"' EXIT
curl -fsS --max-time 10 "${LLM_BASE_URL}/models" >"${LLM_MODELS_TMP}" || {
  echo "[mab] LLM endpoint is not reachable: ${LLM_BASE_URL}" >&2
  cat "${LLM_MODELS_TMP}" 2>/dev/null || true
  exit 3
}
curl -fsS --max-time 10 "${EMBEDDING_BASE_URL}/models" >"${EMBED_MODELS_TMP}" || {
  echo "[mab] embedding endpoint is not reachable: ${EMBEDDING_BASE_URL}" >&2
  cat "${EMBED_MODELS_TMP}" 2>/dev/null || true
  exit 4
}

export OPENAI_BASE_URL="${LLM_BASE_URL}"
export OPENAI_API_KEY
export EMBEDDING_BASE_URL
export EMBEDDING_API_KEY
export QWEN_BASE_URL="${LLM_BASE_URL}"
export QWEN_MODEL_NAME="${LLM_MODEL}"
export VLLM_QWEN_API_KEY="${OPENAI_API_KEY}"
export PYTHONPATH="${REPO_ROOT}/vendor/lightmem/src:${REPO_ROOT}/vendor/memrl:${REPO_ROOT}:${PYTHONPATH:-}"

summary_log="${LOG_DIR}/_summary.log"
echo "[mab] start $(date -Iseconds)" | tee -a "${summary_log}"

for method in ${METHODS}; do
  cell_dir="${RESULTS_ROOT}/${method}"
  log="${LOG_DIR}/${method}.log"
  mkdir -p "${cell_dir}"
  echo "[mab] ===== $(date -Iseconds) ${method} =====" | tee -a "${summary_log}"
  set +e
  method_env=()
  if [[ "${method}" == "memt" ]]; then
    method_env+=(MEMT_MAX_TOKENS="${MEMT_MAX_TOKENS:-256}")
  fi
  env "${method_env[@]}" "${PIXI_PY}" examples/run_benchmark.py \
    --benchmark memoryagentbench \
    --method "${method}" \
    --test-file "${DATASET}" \
    --mab-manifest "${MANIFEST}" \
    --mab-categories AR TTL LRU CR \
    --llm-model "${LLM_MODEL}" \
    --llm-base-url "${LLM_BASE_URL}" \
    --llm-api-key "${OPENAI_API_KEY}" \
    --embedding-model "${EMBED_MODEL}" \
    --embedding-base-url "${EMBEDDING_BASE_URL}" \
    --embedding-api-key "${EMBEDDING_API_KEY}" \
    --output-dir "${cell_dir}" \
    --max-questions "${MAX_QUESTIONS}" \
    --track-efficiency \
    --fail-fast \
    >"${log}" 2>&1
  rc=$?
  set -e
  summary_count="$(find "${cell_dir}" -type f -name summary.json | wc -l | tr -d ' ')"
  if (( rc != 0 )) || [[ "${summary_count}" == "0" ]]; then
    echo "[mab] WARN ${method} rc=${rc}" | tee -a "${summary_log}"
    if [[ "${summary_count}" == "0" ]]; then
      echo "[mab]   no summary.json produced under ${cell_dir}" | tee -a "${summary_log}"
    fi
    tail -30 "${log}" | sed 's/^/[mab]   /' | tee -a "${summary_log}" || true
  else
    echo "[mab] OK ${method} summaries=${summary_count}" | tee -a "${summary_log}"
  fi
done

echo "[mab] done $(date -Iseconds)" | tee -a "${summary_log}"
