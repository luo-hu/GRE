# ALEX-Adaptive 设计说明

这份文档说明 GRE 中新增的 `alexadaptive` 实验变体。它保留 `alex` 作为原始 baseline，只在 `--index=alexadaptive` 时开启动态 gap、误差感知触发和局部重训。

## 核心思路

ALEX 的叶节点用线性模型预测 key 在数组中的位置，并通过 gap 减少插入移动。原版策略的 gap 密度主要来自固定阈值；当局部 CDF 误差或插入移动代价升高时，常见后果是扩容、重训或 split。

`alexadaptive` 做三件事：

1. 动态 gap：叶节点 resize 时不再固定使用 `kMinDensity_`，而是根据当前插入移动、指数搜索步数、append-mostly 行为选择目标密度。
2. 局部重训：当某个叶节点的观测代价超过预期时，优先只重训当前叶节点，并重新铺 gap。
3. 误差感知触发：触发条件来自 ALEX 已有的在线统计，不额外采样，主要包括 `shifts_per_insert()`、`exp_search_iterations_per_operation()` 和 `empirical_cost() / cost_`。

当前版本不是所有 leaf 都启用动态 gap。`--index=alexadaptive` 只是打开实验能力，真正执行前还会调用 `local_hardness_high()` 判断当前 leaf 是否足够困难：

```text
普通 leaf:
  继续使用原版 ALEX 的 kMinDensity_ resize 策略。

难 leaf:
  使用 adaptive_target_density() 选择更适合的 gap 密度，
  必要时触发局部重训。
```

这样做是为了避免 easy 数据上无意义的动态 gap 开销。判定信号来自 ALEX 已有统计，包括：

```text
expected_avg_shifts_
expected_avg_exp_search_iterations_
shifts_per_insert()
exp_search_iterations_per_operation()
empirical_cost() / cost_
```

此外还补了两个更直接的 local hardness 指标，避免局部坏点被平均值抹平：

```text
max_packed_region:
  当前 leaf 内最长的连续无 gap 区域。区域越长，插入越容易产生长距离移动。

max_log_model_error:
  当前 leaf 内模型预测位置与真实位置的最大 log2 误差。误差越大，查询越容易走长距离指数搜索。
```

## 新增索引名

```bash
--index=alexadaptive
```

建议始终和原版 ALEX 成对运行：

```bash
--index=alex,alexadaptive
```

## 新增 CSV 统计列

```text
num_dynamic_gap_resizes
num_error_triggered_retrains
num_adaptive_retrains
```

含义：

```text
num_dynamic_gap_resizes:
  使用自适应目标密度完成的叶节点 resize 次数。

num_error_triggered_retrains:
  因模型误差、搜索步数或插入移动过高而触发的局部叶节点重训次数。

num_adaptive_retrains:
  自适应策略参与的局部重训总次数，包括 split 决策选择“不分裂，只重训”时的情况。
```

如果 `num_dynamic_gap_resizes > 0`，说明动态 gap 已经参与实验。如果 `num_error_triggered_retrains` 长期为 0，说明当前数据和 workload 没有触发误差阈值，可以提高插入比例、使用更偏斜数据，或降低代码中的触发阈值做 sensitivity study。

如果 `alexadaptive` 在某个数据集上 `num_dynamic_gap_resizes = 0`，这表示该 workload 下没有 leaf 被判定为 hard leaf；此时它应当接近原版 ALEX 行为。

## Smoke Test

```bash
./build/microbench \
  --keys_file=./data/smoke_100k_uint64 \
  --keys_file_type=binary \
  --read=0.5 --insert=0.5 \
  --operations_num=2000 \
  --table_size=-1 \
  --init_table_ratio=0.5 \
  --thread_num=1 \
  --index=alexadaptive \
  --latency_sample \
  --latency_sample_ratio=0.1 \
  --memory \
  --output_path=/tmp/gre_alexadaptive_smoke.csv
```

对照：

```bash
./build/microbench \
  --keys_file=./data/smoke_100k_uint64 \
  --keys_file_type=binary \
  --read=0.5 --insert=0.5 \
  --operations_num=2000 \
  --table_size=-1 \
  --init_table_ratio=0.5 \
  --thread_num=1 \
  --index=alex \
  --latency_sample \
  --latency_sample_ratio=0.1 \
  --memory \
  --output_path=/tmp/gre_alex_baseline_smoke.csv
```

## 后续实验建议

优先观察尾延迟和结构调整次数，而不是只看平均吞吐：

```text
99 percentile
99.9 percentile
99.99 percentile
latency_variance
memory_consumption
num_dynamic_gap_resizes
num_error_triggered_retrains
num_expand_and_retrains
num_downward_splits
num_sideways_splits
```

如果动态 gap 显著增加但内存开销过高，可以把 `adaptive_target_density()` 中的低密度阈值从 `0.48/0.50` 调高到 `0.55` 附近。如果误差触发很少，可以把 `should_adaptive_retrain()` 中的 `1.25 * cost_` 或搜索步数阈值下调。
