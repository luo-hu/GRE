#!/usr/bin/env bash
set -Eeuo pipefail

# 批量运行 ALEX-Adaptive 实验矩阵。
# 默认覆盖三类 generator 数据、两个 ALEX 版本、三个随机种子、三组 read/insert workload。
# 结果会追加写入同一个 CSV，便于后续用 pandas/R 统一分析。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MICROBENCH="${MICROBENCH:-./build/microbench}"
GENERATOR="${GENERATOR:-./build/generator}"

# 运行规模。论文级实验建议把 OPERATIONS_NUM 调到 1000000 或更高；
# 调参/冒烟测试可以用较小值，例如 OPERATIONS_NUM=20000。
OPERATIONS_NUM="${OPERATIONS_NUM:-1000000}"
INIT_TABLE_RATIO="${INIT_TABLE_RATIO:-0.5}"
TABLE_SIZE="${TABLE_SIZE:--1}"
THREADS="${THREADS:-1}"
LATENCY_SAMPLE_RATIO="${LATENCY_SAMPLE_RATIO:-0.01}"
KEYS_FILE_TYPE="${KEYS_FILE_TYPE:-binary}"

# 默认只比较原版 ALEX 和本次新增的 alexadaptive。
# 需要更多 baseline 时可以这样运行：
#   INDICES="alex alexadaptive alexol btreeolc lipp" bash run/run_alexadaptive_matrix.sh
INDICES="${INDICES:-alex alexadaptive}"
SEEDS="${SEEDS:-1866 1867 1868}"
WORKLOADS="${WORKLOADS:-0.9:0.1 0.5:0.5 0.1:0.9}"

# 是否在运行前重新生成三份数据。默认不重生成，避免覆盖已有数据。
# 需要重新生成时：
#   GENERATE_DATA=1 bash run/run_alexadaptive_matrix.sh
GENERATE_DATA="${GENERATE_DATA:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-results/alexadaptive_matrix/${RUN_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-${RESULT_DIR}/matrix.csv}"
LOG_PATH="${LOG_PATH:-${RESULT_DIR}/run.log}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${RESULT_DIR}"

DATASETS=(
  "easy:data/alexrobust/synth_easy_1M:64:64:1:1:1000000"
  "local:data/alexrobust/synth_local_1M:64:64:1000:1:1000000"
  "both:data/alexrobust/synth_both_1M:64:64:1000:1000:1000000"
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_PATH}"
}

run_cmd() {
  log "$*"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@" 2>&1 | tee -a "${LOG_PATH}"
  fi
}

check_binary() {
  local path="$1"
  local target="$2"
  if [[ -x "${path}" ]]; then
    return
  fi
  log "${path} 不存在或不可执行，尝试编译 ${target}"
  run_cmd cmake --build build --target "${target}" -j"$(nproc)"
}

generate_datasets_if_needed() {
  if [[ "${GENERATE_DATA}" != "1" ]]; then
    return
  fi

  check_binary "${GENERATOR}" generator
  mkdir -p data/alexrobust

  for item in "${DATASETS[@]}"; do
    IFS=':' read -r name path x y local_hard global_hard count <<< "${item}"
    log "生成数据集 ${name}: ${path}"
    run_cmd "${GENERATOR}" "${x}" "${y}" "${local_hard}" "${global_hard}" "${count}" "${path}"
  done
}

validate_datasets() {
  for item in "${DATASETS[@]}"; do
    IFS=':' read -r name path _ <<< "${item}"
    if [[ ! -f "${path}" ]]; then
      log "缺少数据集 ${name}: ${path}"
      log "可以先运行 GENERATE_DATA=1 bash run/run_alexadaptive_matrix.sh"
      exit 1
    fi
  done
}

main() {
  : > "${LOG_PATH}"
  log "结果目录: ${RESULT_DIR}"
  log "CSV 输出: ${OUTPUT_PATH}"
  log "索引: ${INDICES}"
  log "随机种子: ${SEEDS}"
  log "workload(read:insert): ${WORKLOADS}"
  log "线程数: ${THREADS}"
  log "每次 operations_num: ${OPERATIONS_NUM}"

  check_binary "${MICROBENCH}" microbench
  generate_datasets_if_needed
  validate_datasets

  local total=0
  for _dataset in "${DATASETS[@]}"; do
    for _index in ${INDICES}; do
      for _seed in ${SEEDS}; do
        for _workload in ${WORKLOADS}; do
          for _thread in ${THREADS}; do
            total=$((total + 1))
          done
        done
      done
    done
  done
  log "计划运行 ${total} 组实验"

  local current=0
  for dataset in "${DATASETS[@]}"; do
    IFS=':' read -r dataset_name keys_file _ <<< "${dataset}"
    for index in ${INDICES}; do
      for seed in ${SEEDS}; do
        for workload in ${WORKLOADS}; do
          IFS=':' read -r read_ratio insert_ratio <<< "${workload}"
          for thread_num in ${THREADS}; do
            current=$((current + 1))
            log "(${current}/${total}) dataset=${dataset_name} index=${index} seed=${seed} read=${read_ratio} insert=${insert_ratio} threads=${thread_num}"
            run_cmd "${MICROBENCH}" \
              --keys_file="${keys_file}" \
              --keys_file_type="${KEYS_FILE_TYPE}" \
              --read="${read_ratio}" \
              --insert="${insert_ratio}" \
              --operations_num="${OPERATIONS_NUM}" \
              --table_size="${TABLE_SIZE}" \
              --init_table_ratio="${INIT_TABLE_RATIO}" \
              --thread_num="${thread_num}" \
              --index="${index}" \
              --seed="${seed}" \
              --latency_sample \
              --latency_sample_ratio="${LATENCY_SAMPLE_RATIO}" \
              --memory \
              --output_path="${OUTPUT_PATH}"
          done
        done
      done
    done
  done

  log "全部完成: ${OUTPUT_PATH}"
}

main "$@"
