# ALEX-Robust 设计说明

这份文档解释 GRE 里新增的 `alexrobust` 做了什么、为什么这样做，以及应该怎么验证它有没有效果。

## 1. 先理解项目里的两层 ALEX 代码

GRE 里和 ALEX 有关的代码主要分两层：

1. `src/competitor/alex/alex.h`

   这是 GRE benchmark 的包装层。GRE 希望所有索引都长得一样，都能调用 `bulk_load()`、`get()`、`put()`、`scan()`。原始 ALEX 的接口和 GRE 不完全一样，所以这里负责把 ALEX 包装成 GRE 能调用的形式。

2. `src/competitor/alex/src/src/core/alex.h` 和 `alex_nodes.h`

   这是 ALEX 原始实现的核心层。`alex.h` 负责整棵树的 bulk load、insert、split、统计信息；`alex_nodes.h` 负责节点，尤其是 data node，也就是叶节点。

简单说：

```text
microbench
  -> GRE 包装层 src/competitor/alex/alex.h
    -> ALEX 核心层 src/competitor/alex/src/src/core/alex.h
      -> ALEX 叶节点 src/competitor/alex/src/src/core/alex_nodes.h
```

## 2. alexrobust 注册是什么意思

GRE 通过命令行参数选择索引：

```bash
--index=alex
```

这个字符串会传给 `src/competitor/competitor.h` 里的 `get_index()`。所谓“注册 alexrobust”，只是让这个函数认识一个新名字：

```text
--index=alex       -> 创建原版 ALEX
--index=alexrobust -> 创建启用 robust leaf 的 ALEX
```

它不是安装系统组件，也不是修改环境变量，只是给 benchmark 增加了一个可选实验对象。

## 3. ALEX 叶节点原本怎么工作

ALEX 是 learned index。它会学习一个模型，大概预测：

```text
这个 key 应该在叶节点数组的哪个位置？
```

叶节点里不是紧凑数组，而是带空洞的数组。比如：

```text
10 _ _ 20 _ 30 _ _ 40
```

这些空洞的好处是插入可能更快。比如插入 25，可以找附近空洞放进去，不一定要移动大量元素。

但坏处是，如果局部模型预测不准，或者空洞分布很糟，查找和插入都可能突然变慢。平均吞吐可能还行，但 P99/P99.9 这种尾延迟会很难看。

## 4. alexrobust 的核心想法

`alexrobust` 不追求让每个叶节点都继续相信模型。它的第一版只做一件事：

```text
当某个 ALEX 叶节点看起来风险很高时，把这个叶节点转换成紧凑有序数组。
```

转换前：

```text
10 _ _ 20 _ 30 _ _ 40
模型预测 + 指数搜索 + 找空洞插入
```

转换后：

```text
10 20 30 40
二分查找 + 普通数组移动插入
```

这就是 robust leaf mode。它更像一个小型稳定数组叶节点。它不一定在平均性能上更激进，但行为更容易解释，最坏情况也更可控。

## 5. 当前版本的风险判断

风险判断在 `AlexDataNode::robust_risk_high()` 里。现在用的是很朴素的启发式：

```text
如果 robust leaf 功能没有开启：
    不转换
如果这个叶节点已经是 robust leaf：
    不重复转换
如果这个叶节点太小：
    不转换
如果查找搜索步数太高：
    转换
如果插入平均移动次数太高：
    转换
如果 ALEX 预估插入移动代价太高：
    转换
```

对应代码里的指标大致是：

```text
exp_search_iterations_per_operation()
shifts_per_insert()
expected_avg_shifts_
```

这些指标都来自 ALEX 原本已有的统计逻辑，不是额外做复杂训练。

## 6. 为什么不是直接修改 alex

为了做实验对照，原版 `alex` 保持默认不开启 robust leaf。新增 `alexrobust` 的好处是可以直接跑：

```bash
--index=alex
--index=alexrobust
```

然后比较两者：

```text
吞吐
P99/P99.9
内存
split / retrain / expand 次数
robust leaf 转换次数
```

这样论文实验才有清楚的 baseline 和 ablation。

## 7. 新增 CSV 统计列

`alexrobust` 新增了两个诊断列：

```text
num_robust_leaf_conversions
num_robust_leaf_inserts
```

含义：

```text
num_robust_leaf_conversions:
  有多少个叶节点被转换成 robust leaf。

num_robust_leaf_inserts:
  有多少次插入是在 robust leaf 里完成的。
```

如果 P99 下降，但这两个数一直是 0，说明本次实验没有真正触发 robust leaf，不能说改动产生了效果。

如果这两个数上升，同时 P99/P99.9 更稳定，就说明“高风险叶节点兜底”开始发挥作用。

## 8. 如何验证效果

验证分三步：正确性、触发情况、性能效果。

### 8.1 正确性 smoke test

先用小数据确认程序不崩，插入和查询结果数量合理：

```bash
seq 1 10000 > /tmp/gre_keys.txt

./build/microbench \
  --keys_file=/tmp/gre_keys.txt \
  --keys_file_type=text \
  --read=0.5 --insert=0.5 \
  --operations_num=1000 \
  --table_size=10000 \
  --init_table_ratio=0.5 \
  --thread_num=1 \
  --index=alexrobust \
  --latency_sample \
  --latency_sample_ratio=0.1 \
  --memory \
  --output_path=/tmp/gre_alexrobust_smoke.csv
```

看输出里有没有：

```text
Finish running
success_read
success_insert
```

再看 CSV 是否有新增列：

```bash
head -n 1 /tmp/gre_alexrobust_smoke.csv
```

### 8.2 对照实验

同一份数据、同一个 seed、同一组 workload，分别跑：

```bash
--index=alex
--index=alexrobust
```

建议输出到两个 CSV：

```bash
--output_path=results/alex_tail.csv
--output_path=results/alexrobust_tail.csv
```

重点比较这些列：

```text
throughput
memory_consumption
99 percentile
99.9 percentile
99.99 percentile
latency_variance
num_expand_and_scales
num_expand_and_retrains
num_downward_splits
num_sideways_splits
num_robust_leaf_conversions
num_robust_leaf_inserts
```

### 8.3 怎么判断“有效”

不是只看 throughput。这个方向的目标是鲁棒性和尾延迟，所以更推荐这样判断：

```text
如果 alexrobust 的 P99/P99.9 更低或波动更小，
同时 throughput 没有严重下降，
并且 num_robust_leaf_conversions > 0，
那说明 robust leaf 有初步效果。
```

如果结果是：

```text
num_robust_leaf_conversions = 0
```

说明当前数据或 workload 没触发风险判断。下一步要么换更难的数据/插入模式，要么降低风险阈值做 sensitivity study。

如果结果是：

```text
P99 没改善，但 robust conversions 很多
```

说明转换策略可能太激进，或者 compact leaf 的插入移动成本压过了查找稳定性收益。下一步要调阈值或把 robust leaf 改成 block/box leaf。

## 9. 当前版本的局限

这只是第一阶段原型，还不是完整论文系统。

当前已经有：

```text
原生 ALEX leaf
robust compact leaf
风险触发转换
CSV 诊断统计
```

当前还没有：

```text
hot/cold workload 感知
SIMD box leaf
range-scan dense leaf
自动调参
tail latency debt 的在线采样
```

所以它适合用来验证一个最小问题：

```text
当 ALEX 的局部 learned leaf 风险升高时，退回更稳定的叶结构能不能改善尾延迟？
```

