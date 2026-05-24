#!/usr/bin/env bash
set -Eeuo pipefail

# 批量运行 lipp / lipphybrid 实验矩阵。
# 默认覆盖三类 generator 数据、两个 lipp 版本、三个随机种子、三组 read/insert workload。
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
LATENCY_SAMPLE="${LATENCY_SAMPLE:-1}"
LATENCY_SAMPLE_RATIO="${LATENCY_SAMPLE_RATIO:-0.01}"
KEYS_FILE_TYPE="${KEYS_FILE_TYPE:-binary}"

# 默认只比较原版 lipp 和 lipphybrid 默认策略。
# 验证 sqrt 自适应时可以设置 KERNEL_SWEEP=1，默认只跑当前还值得继续比较的版本：
#   KERNEL_SWEEP=1 bash run/run_lipp_matrix.sh
# 如果还需要 log1p/cbrt 的完整消融，可以额外设置：
#   FULL_KERNEL_SWEEP=1 bash run/run_lipp_matrix.sh
# 验证紧凑叶节点时可以设置：
#   LEAF_SWEEP=1 bash run/run_lipp_matrix.sh
# 其中 *_compact 是强制小叶子紧凑化，*_adaptive_compact 是按局部成本模型自适应开启。
# 也可以手动指定：
#   INDICES="lipp lipphybrid_identity lipphybrid_aggressive lipphybrid_sqrt" bash run/run_lipp_matrix.sh
KERNEL_SWEEP="${KERNEL_SWEEP:-0}"
FULL_KERNEL_SWEEP="${FULL_KERNEL_SWEEP:-0}"
LEAF_SWEEP="${LEAF_SWEEP:-0}"
if [[ "${FULL_KERNEL_SWEEP}" == "1" && -z "${INDICES:-}" ]]; then
  INDICES="lipp lipphybrid_identity lipphybrid_fallback lipphybrid_aggressive lipphybrid_log1p lipphybrid_sqrt lipphybrid_cbrt"
elif [[ "${LEAF_SWEEP}" == "1" && -z "${INDICES:-}" ]]; then
  INDICES="lipp lipphybrid_identity lipphybrid_compact lipphybrid_adaptive_compact lipphybrid_aggressive lipphybrid_aggressive_compact lipphybrid_aggressive_adaptive_compact"
elif [[ "${KERNEL_SWEEP}" == "1" && -z "${INDICES:-}" ]]; then
  INDICES="lipp lipphybrid_identity lipphybrid_fallback lipphybrid_aggressive lipphybrid_sqrt"
else
  INDICES="${INDICES:-lipp lipphybrid}"
fi
SEEDS="${SEEDS:-1866 1867 1868}"
WORKLOADS="${WORKLOADS:-0.9:0.1 0.5:0.5 0.1:0.9}"

# 是否在运行前重新生成三份数据。默认不重生成，避免覆盖已有数据。
# 需要重新生成时：
#   GENERATE_DATA=1 bash run/run_lipp_matrix.sh
GENERATE_DATA="${GENERATE_DATA:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-results/lipp_matrix/${RUN_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-${RESULT_DIR}/matrix.csv}"
LOG_PATH="${LOG_PATH:-${RESULT_DIR}/run.log}"
DRY_RUN="${DRY_RUN:-0}"

# 这份 OSM Antarctica 文件通常是“纯 uint64 key 流”，没有 GRE 二进制格式要求的数量头。
# GRE 的 load_binary_data() 会把第一个 uint64 当作 key 数量，所以这里默认转换成 header+keys 格式。
OSM_RAW_PATH="${OSM_RAW_PATH:-data/alexrobust/osm_antarctica_1m.bin}"
OSM_GRE_PATH="${OSM_GRE_PATH:-data/alexrobust/osm_antarctica_1m.gre.bin}"
# 新增 500万 版本
OSM_5M_RAW_PATH="${OSM_5M_RAW_PATH:-data/alexrobust/osm_antarctica_5m.bin}"
OSM_5M_GRE_PATH="${OSM_5M_GRE_PATH:-data/alexrobust/osm_antarctica_5m.gre.bin}"
# 5M OSM 数据集体积更大，默认 auto：有 raw/gre 文件就加入实验，没有就跳过。
# 可设置 INCLUDE_OSM5M=0 强制跳过，或 INCLUDE_OSM5M=1 要求必须存在。
INCLUDE_OSM5M="${INCLUDE_OSM5M:-auto}"

mkdir -p "${RESULT_DIR}"

DATASETS=(
  "easy:data/alexrobust/synth_easy_1M:64:64:1:1:1000000"
  "local:data/alexrobust/synth_local_1M:64:64:1000:1:1000000"
  "both:data/alexrobust/synth_both_1M:64:64:1000:1000:1000000"
  "osm:${OSM_GRE_PATH}"
)

if [[ "${INCLUDE_OSM5M}" == "1" || ( "${INCLUDE_OSM5M}" == "auto" && ( -f "${OSM_5M_RAW_PATH}" || -f "${OSM_5M_GRE_PATH}" ) ) ]]; then
  DATASETS+=("osm5m:${OSM_5M_GRE_PATH}")
fi

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
  mkdir -p data/lipprobust

  for item in "${DATASETS[@]}"; do
    IFS=':' read -r name path x y local_hard global_hard count <<< "${item}"
    if [[ -z "${count:-}" ]]; then
      continue
    fi
    log "生成数据集 ${name}: ${path}"
    run_cmd "${GENERATOR}" "${x}" "${y}" "${local_hard}" "${global_hard}" "${count}" "${path}"
  done
}

prepare_uint64_gre_dataset() {
  local name="$1"
  local raw_path="$2"
  local gre_path="$3"
  local required="$4"

  if [[ ! -f "${raw_path}" ]]; then
    if [[ -f "${gre_path}" ]]; then
      # 已经有转换后的 GRE 文件时，不再要求 raw 文件存在。
      return
    fi
    if [[ "${required}" == "1" ]]; then
      log "缺少 ${name} 原始数据集: ${raw_path}"
      log "也可以通过 ${name}_RAW_PATH=/path/to/file 覆盖路径"
      exit 1
    fi
    log "跳过可选数据集 ${name}: 未找到 ${raw_path} 或 ${gre_path}"
    return
  fi

  if [[ -f "${gre_path}" && "${gre_path}" -nt "${raw_path}" ]]; then
    return
  fi

  log "转换 ${name} 数据为 GRE 二进制格式: ${raw_path} -> ${gre_path}"
  mkdir -p "$(dirname "${gre_path}")"
  python3 - "${raw_path}" "${gre_path}" <<'PY'
import os
import shutil
import struct
import sys

src, dst = sys.argv[1], sys.argv[2]
size = os.path.getsize(src)
if size == 0 or size % 8 != 0:
    raise SystemExit(f"invalid uint64 binary file size: {src} ({size} bytes)")

with open(src, "rb") as f:
    first = struct.unpack("<Q", f.read(8))[0]

# 如果输入本来就是 GRE 格式（第一个 uint64 是数量头），直接复制。
if size == (first + 1) * 8:
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
else:
    # 否则把整个文件视为 raw uint64 key 数组，并在前面补 key 数量。
    count = size // 8
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<Q", count))
        shutil.copyfileobj(fin, fout)
PY
}

prepare_osm_dataset() {
  prepare_uint64_gre_dataset "OSM" "${OSM_RAW_PATH}" "${OSM_GRE_PATH}" 1

  if [[ "${INCLUDE_OSM5M}" == "0" ]]; then
    return
  fi

  local require_osm5m=0
  if [[ "${INCLUDE_OSM5M}" == "1" ]]; then
    require_osm5m=1
  fi
  prepare_uint64_gre_dataset "OSM_5M" "${OSM_5M_RAW_PATH}" "${OSM_5M_GRE_PATH}" "${require_osm5m}"
}

validate_datasets() {
  for item in "${DATASETS[@]}"; do
    IFS=':' read -r name path _ <<< "${item}"
    if [[ ! -f "${path}" ]]; then
      log "缺少数据集 ${name}: ${path}"
      log "可以先运行 GENERATE_DATA=1 bash run/run_lipp_matrix.sh"
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
  log "latency_sample: ${LATENCY_SAMPLE}"

  check_binary "${MICROBENCH}" microbench
  generate_datasets_if_needed
  prepare_osm_dataset
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
            cmd=(
              "${MICROBENCH}"
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
              --memory \
              --output_path="${OUTPUT_PATH}"
            )
            if [[ "${LATENCY_SAMPLE}" == "1" ]]; then
              cmd+=(--latency_sample --latency_sample_ratio="${LATENCY_SAMPLE_RATIO}")
            fi
            run_cmd "${cmd[@]}"
          done
        done
      done
    done
  done

  log "全部完成: ${OUTPUT_PATH}"
}

main "$@"
