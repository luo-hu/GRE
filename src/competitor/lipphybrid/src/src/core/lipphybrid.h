#ifndef __LIPPHYBRID_H__
#define __LIPPHYBRID_H__

#include "lipphybrid_base.h"
#include <stdint.h>
#include <math.h>
#include <limits>
#include <cstdio>
#include <stack>
#include <vector>
#include <cstring>
#include <sstream>
#include <algorithm>

typedef uint8_t bitmap_t;
#define BITMAP_WIDTH (sizeof(bitmap_t) * 8)
#define BITMAP_SIZE(num_items) (((num_items) + BITMAP_WIDTH - 1) / BITMAP_WIDTH)
#define BITMAP_GET(bitmap, pos) (((bitmap)[(pos) / BITMAP_WIDTH] >> ((pos) % BITMAP_WIDTH)) & 1)
#define BITMAP_SET(bitmap, pos) ((bitmap)[(pos) / BITMAP_WIDTH] |= 1 << ((pos) % BITMAP_WIDTH))
#define BITMAP_CLEAR(bitmap, pos) ((bitmap)[(pos) / BITMAP_WIDTH] &= ~bitmap_t(1 << ((pos) % BITMAP_WIDTH)))
#define BITMAP_NEXT_1(bitmap_item) __builtin_ctz((bitmap_item))

// runtime assert
#define LIPPHYBRID_RT_ASSERT(expr) \
{ \
    if (!(expr)) { \
        fprintf(stderr, "LIPPHYBRID_RT_ASSERT Error at %s:%d, `%s`\n", __FILE__, __LINE__, #expr); \
        exit(0); \
    } \
}

#define COLLECT_TIME 0

#if COLLECT_TIME
#include <chrono>
#endif

enum class LIPPHybridKernelPolicy {
    IDENTITY_ONLY = 0,          // 完全等价于 identity 模型，用来做基线消融。
    FALLBACK_ONLY = 1,          // 当前默认策略：FMCD 成功时不动，只在 fallback 中尝试核函数。
    ADAPTIVE_AGGRESSIVE = 2,    // 更激进：FMCD 主路径和 fallback 都允许选择非线性核。
    FORCE_LOG1P = 3,            // 强制使用 log1p 核，用来观察该核函数的上限/风险。
    FORCE_SQRT = 4,             // 强制使用 sqrt 核。
    FORCE_CBRT = 5              // 强制使用 cbrt 核。
};

enum class LIPPHybridCompactPolicy {
    DISABLED = 0,               // 完全不使用紧凑叶节点，作为 LIPP/Fmcd 精确槽位基线。
    FORCE = 1,                  // 只要叶子足够小就强制紧凑化，用来观察 compact leaf 的收益和风险。
    ADAPTIVE = 2                // 按局部成本模型自动决定是否紧凑化，不再针对某个数据集手调阈值。
};

template<class T, class P, bool USE_FMCD = true>
class LIPPHybrid
{
    static_assert(std::is_arithmetic<T>::value, "LIPP key type must be numeric.");

    inline int compute_gap_count(int size) const {
        if (size >= 1000000) return 1;
        if (size >= 100000) return 2;
        return 5;
    }

    struct Node;
    typedef typename HybridLinearModel<T>::Kernel Kernel;

    inline int PREDICT_POS(Node* node, T key) const {
        double v = node->model.predict_double(key);
        if (v > std::numeric_limits<int>::max() / 2) {
            return node->num_items - 1;
        }
        if (v < 0) {
            return 0;
        }
        return std::min(node->num_items - 1, static_cast<int>(v));
    }

    static void remove_last_bit(bitmap_t& bitmap_item) {
        bitmap_item -= 1 << BITMAP_NEXT_1(bitmap_item);
    }

    static int kernel_compute_penalty(Kernel kernel) {
        // 非线性核函数能降低冲突，但预测时会多一次数学函数调用。
        // 给它们一个很小的惩罚，避免在收益不明显时误选复杂核。
        switch (kernel) {
        case HybridLinearModel<T>::LOG1P:
            return 40;
        case HybridLinearModel<T>::SQRT:
            return 30;
        case HybridLinearModel<T>::CBRT:
            return 45;
        case HybridLinearModel<T>::IDENTITY:
        default:
            return 0;
        }
    }

    long long estimate_model_cost(T* keys, int size, int num_items,
                                  const HybridLinearModel<T>& model,
                                  Kernel kernel) const {
        // 只抽样估算冲突度，避免根节点上为了选核函数额外扫描/分配巨大数组。
        // 后续若要写论文实验，可以把 sample_limit 做成参数。
        const int sample_limit = 4096;
        const int step = std::max(1, size / sample_limit);
        std::vector<int> slots;
        slots.reserve(std::min(size, sample_limit + 2));
        for (int i = 0; i < size; i += step) {
            double v = model.predict_double(keys[i]);
            int pos;
            if (v > std::numeric_limits<int>::max() / 2) {
                pos = num_items - 1;
            } else if (v < 0) {
                pos = 0;
            } else {
                pos = std::min(num_items - 1, static_cast<int>(v));
            }
            slots.push_back(pos);
        }
        if (slots.empty() || slots.back() != num_items - 1) {
            double v = model.predict_double(keys[size - 1]);
            int pos = v < 0 ? 0 : std::min(num_items - 1, static_cast<int>(v));
            slots.push_back(pos);
        }

        std::sort(slots.begin(), slots.end());
        long long conflict_slots = 0;
        long long conflict_keys = 0;
        long long max_bucket = 1;
        for (size_t i = 0; i < slots.size();) {
            size_t j = i + 1;
            while (j < slots.size() && slots[j] == slots[i]) {
                j++;
            }
            const long long bucket = static_cast<long long>(j - i);
            if (bucket > 1) {
                conflict_slots++;
                conflict_keys += bucket - 1;
                max_bucket = std::max(max_bucket, bucket);
            }
            i = j;
        }

        return conflict_keys * 1000 + conflict_slots * 100 + max_bucket * 10 + kernel_compute_penalty(kernel);
    }

    long long estimate_midpoint_kernel_cost(T* keys, int size, int num_items, Kernel kernel,
                                            int mid1_pos, int mid2_pos,
                                            double mid1_target, double mid2_target) const {
        HybridLinearModel<T> model;
        model.set_kernel(kernel, keys[0]);

        const long double mid1_key =
            (static_cast<long double>(keys[mid1_pos]) + static_cast<long double>(keys[mid1_pos + 1])) / 2;
        const long double mid2_key =
            (static_cast<long double>(keys[mid2_pos]) + static_cast<long double>(keys[mid2_pos + 1])) / 2;
        const long double tx1 = model.transform_value(mid1_key);
        const long double tx2 = model.transform_value(mid2_key);
        if (!(tx2 > tx1)) {
            return std::numeric_limits<long long>::max() / 4;
        }

        model.a = (mid2_target - mid1_target) / (tx2 - tx1);
        model.b = mid1_target - model.a * tx1;
        if (!isfinite(model.a) || !isfinite(model.b) || model.a < 0) {
            return std::numeric_limits<long long>::max() / 4;
        }

        return estimate_model_cost(keys, size, num_items, model, kernel);
    }

    long long estimate_fmcd_kernel_cost(T* keys, int size, int num_items, Kernel kernel) const {
        HybridLinearModel<T> model;
        model.set_kernel(kernel, keys[0]);

        // FMCD 主路径最终不是使用三分位点拟合，而是用 D/Ut 推导斜率。
        // 这里按同样逻辑轻量模拟一遍，否则自适应选择器评估的模型会和实际建树模型不一致。
        int i = 0;
        int D = 1;
        if (size <= 2 || num_items <= 2 || !(D <= size - 1 - D)) {
            return std::numeric_limits<long long>::max() / 4;
        }

        double Ut = (model.transform(keys[size - 1 - D]) - model.transform(keys[D])) /
                    (static_cast<double>(num_items - 2)) + 1e-6;
        while (i < size - 1 - D) {
            while (i + D < size && model.transform(keys[i + D]) - model.transform(keys[i]) >= Ut) {
                i++;
            }
            if (i + D >= size) {
                break;
            }
            D = D + 1;
            if (D * 3 > size) {
                break;
            }
            if (!(D <= size - 1 - D)) {
                return std::numeric_limits<long long>::max() / 4;
            }
            Ut = (model.transform(keys[size - 1 - D]) - model.transform(keys[D])) /
                 (static_cast<double>(num_items - 2)) + 1e-6;
        }

        if (D * 3 > size || !(Ut > 0)) {
            return std::numeric_limits<long long>::max() / 4;
        }

        model.a = 1.0 / Ut;
        model.b = (num_items - model.a * (model.transform(keys[size - 1 - D]) +
                                          model.transform(keys[D]))) / 2;
        if (!isfinite(model.a) || !isfinite(model.b) || model.a < 0) {
            return std::numeric_limits<long long>::max() / 4;
        }

        return estimate_model_cost(keys, size, num_items, model, kernel);
    }

    Kernel forced_kernel() const {
        if (kernel_policy == LIPPHybridKernelPolicy::FORCE_LOG1P) {
            return HybridLinearModel<T>::LOG1P;
        }
        if (kernel_policy == LIPPHybridKernelPolicy::FORCE_SQRT) {
            return HybridLinearModel<T>::SQRT;
        }
        if (kernel_policy == LIPPHybridKernelPolicy::FORCE_CBRT) {
            return HybridLinearModel<T>::CBRT;
        }
        return HybridLinearModel<T>::IDENTITY;
    }

    struct SqrtHardnessSummary {
        long double avg_gap = 0;
        long double p50_gap = 0;
        long double p99_gap = 0;
        long double gap_tail_ratio = 0;
        int build_gap_count = 0;
    };

    SqrtHardnessSummary summarize_sqrt_hardness(T* keys, int size) {
        SqrtHardnessSummary summary;
        if (size < 2) {
            return summary;
        }

        const long double span =
            static_cast<long double>(keys[size - 1]) - static_cast<long double>(keys[0]);
        summary.avg_gap = span / static_cast<long double>(size - 1);
        summary.build_gap_count = compute_gap_count(size);

        // GRE 的 hardness 关注“局部难度”和长尾分布，而不只是整体跨度。
        // 这里抽样 gap 分布，用 p99/p50 近似局部密度突变程度：
        // 比值越大，说明一部分区间特别稀疏、一部分区间特别密集，线性路由更容易产生冲突子树。
        const int sample_limit = 8192;
        const int step = std::max(1, (size - 1) / sample_limit);
        std::vector<long double> gaps;
        gaps.reserve(std::min(size - 1, sample_limit + 2));
        for (int i = 0; i + 1 < size; i += step) {
            gaps.push_back(static_cast<long double>(keys[i + 1]) - static_cast<long double>(keys[i]));
        }
        if (gaps.empty()) {
            return summary;
        }

        std::sort(gaps.begin(), gaps.end());
        const size_t p50_pos = gaps.size() / 2;
        const size_t p99_pos = std::min(gaps.size() - 1, static_cast<size_t>(gaps.size() * 99 / 100));
        summary.p50_gap = gaps[p50_pos];
        summary.p99_gap = gaps[p99_pos];
        summary.gap_tail_ratio = summary.p99_gap / std::max<long double>(1.0L, summary.p50_gap);
        return summary;
    }

    bool should_enable_global_sqrt(T* keys, int size) {
        if (kernel_policy != LIPPHybridKernelPolicy::ADAPTIVE_AGGRESSIVE || size < 64) {
            return false;
        }

        const SqrtHardnessSummary hardness = summarize_sqrt_hardness(keys, size);

        // sqrt 在 OSM 1M 上有效，不只是因为 key 空间跨度大，还因为 gap 分布长尾明显。
        // avg_gap 区分“整体稀疏”和普通合成数据；p99/p50 区分“局部密度突变”；
        // build_gap_count 则表示 LIPP 当前节点仍会预留较多空洞，sqrt 才有机会换来结构压缩。
        const bool sparse_key_space = hardness.avg_gap > 1000.0L;
        const bool long_tail_gaps = hardness.gap_tail_ratio > 20.0L;
        const bool lipp_has_gap_budget = hardness.build_gap_count > 1;
        return sparse_key_space && long_tail_gaps && lipp_has_gap_budget;
    }

    static long long ceil_log2_int(int value) {
        long long ans = 0;
        int x = std::max(1, value - 1);
        while (x > 0) {
            ans++;
            x >>= 1;
        }
        return ans;
    }

    static long long estimated_item_bytes() {
        // Item 实际上是 “key/value 数据” 和 “child 指针” 的 union。
        // 这里用对齐后的估算值，避免 should_use_local_adaptive_compact 依赖后面才定义的 Item。
        const long long align = static_cast<long long>(sizeof(void*));
        const long long data_bytes = static_cast<long long>(sizeof(T) + sizeof(P));
        const long long raw_bytes = std::max<long long>(data_bytes, static_cast<long long>(sizeof(void*)));
        return ((raw_bytes + align - 1) / align) * align;
    }

    static long long estimated_node_bytes() {
        // Node 的精确 sizeof 要等到 private 区域里的 Node 定义之后才能用。
        // 这里按字段估算：7 个 int、一个模型、items 指针、两个 bitmap 指针。
        const long long align = static_cast<long long>(sizeof(void*));
        long long raw_bytes = static_cast<long long>(sizeof(int) * 7 +
                              sizeof(HybridLinearModel<T>) +
                              sizeof(void*) * 3);
        return ((raw_bytes + align - 1) / align) * align;
    }

    static long long estimated_bitmap_bytes(int num_items) {
        return static_cast<long long>(BITMAP_SIZE(num_items)) *
               static_cast<long long>(sizeof(bitmap_t)) * 2;
    }

    static int compact_leaf_capacity_for_size(int size) {
        return std::max(8, size * 2);
    }

    long double expected_operations_for_local_size(int local_size, long double op_ratio) const {
        if (workload_total_keys <= 0 || workload_operations_num <= 0 || op_ratio <= 0) {
            return 0;
        }
        return static_cast<long double>(workload_operations_num) * op_ratio *
               static_cast<long double>(local_size) /
               static_cast<long double>(workload_total_keys);
    }

    long double estimated_compact_lookup_units(int size, long long item_bytes) const {
        if (size <= COMPACT_LINEAR_SCAN_MAX_SIZE) {
            // 真实查找路径里，小 compact leaf 使用线性扫描。
            // 线性扫描虽然比较次数可能多于二分，但 key/value 连续存放；
            // 一个 64B cache line 通常能装多个 Item，所以这里按“会跨过几条 cache line”
            // 估算有效访问成本，而不是粗暴按比较次数计费。
            const long double avg_scanned_items = (static_cast<long double>(size) + 1.0L) / 2.0L;
            const long double items_per_cacheline =
                std::max<long double>(1.0L, 64.0L / static_cast<long double>(item_bytes));
            return std::max<long double>(1.0L, avg_scanned_items / items_per_cacheline);
        }

        if (size <= COMPACT_SEGMENT_SCAN_MAX_SIZE) {
            // 中等大小 compact leaf 使用“分段查找”：
            // 先看每段最后一个 key，确定落在哪个小段，再在段内线性扫描。
            // 成本按平均会访问的段尾 key 加半个段内扫描量估算。
            const int blocks = (size + COMPACT_SEGMENT_SIZE - 1) / COMPACT_SEGMENT_SIZE;
            const long double avg_block_probes =
                (static_cast<long double>(blocks) + 1.0L) / 2.0L;
            const long double avg_in_block_scan =
                (static_cast<long double>(COMPACT_SEGMENT_SIZE) + 1.0L) / 2.0L;
            const long double items_per_cacheline =
                std::max<long double>(1.0L, 64.0L / static_cast<long double>(item_bytes));
            return std::max<long double>(1.0L,
                                         (avg_block_probes + avg_in_block_scan) /
                                         items_per_cacheline);
        }

        // 较大的 compact leaf 仍然走二分查找，访问位置跳跃，按二分步数估算。
        return static_cast<long double>(ceil_log2_int(size));
    }

    long double estimated_compact_compare_units(int size) const {
        if (size <= COMPACT_LINEAR_SCAN_MAX_SIZE) {
            return (static_cast<long double>(size) + 1.0L) / 2.0L;
        }
        if (size <= COMPACT_SEGMENT_SCAN_MAX_SIZE) {
            const int blocks = (size + COMPACT_SEGMENT_SIZE - 1) / COMPACT_SEGMENT_SIZE;
            const long double avg_block_probes =
                (static_cast<long double>(blocks) + 1.0L) / 2.0L;
            const long double avg_in_block_scan =
                (static_cast<long double>(COMPACT_SEGMENT_SIZE) + 1.0L) / 2.0L;
            return avg_block_probes + avg_in_block_scan;
        }
        return static_cast<long double>(ceil_log2_int(size));
    }

    bool fit_midpoint_model(T* keys, int size, int num_items, Kernel kernel,
                            HybridLinearModel<T>& model) const {
        const int build_gap_count = compute_gap_count(size);
        const int mid1_pos = (size - 1) / 3;
        const int mid2_pos = (size - 1) * 2 / 3;
        if (!(0 <= mid1_pos && mid1_pos < mid2_pos && mid2_pos < size - 1)) {
            return false;
        }

        const long double mid1_key =
            (static_cast<long double>(keys[mid1_pos]) + static_cast<long double>(keys[mid1_pos + 1])) / 2;
        const long double mid2_key =
            (static_cast<long double>(keys[mid2_pos]) + static_cast<long double>(keys[mid2_pos + 1])) / 2;
        const double mid1_target =
            mid1_pos * static_cast<int>(build_gap_count + 1) + static_cast<int>(build_gap_count + 1) / 2;
        const double mid2_target =
            mid2_pos * static_cast<int>(build_gap_count + 1) + static_cast<int>(build_gap_count + 1) / 2;

        model.set_kernel(kernel, keys[0]);
        const long double tx1 = model.transform_value(mid1_key);
        const long double tx2 = model.transform_value(mid2_key);
        if (!(tx2 > tx1)) {
            return false;
        }

        model.a = (mid2_target - mid1_target) / (tx2 - tx1);
        model.b = mid1_target - model.a * tx1;
        return isfinite(model.a) && isfinite(model.b) && model.a >= 0;
    }

    bool fit_fmcd_model(T* keys, int size, int num_items, Kernel kernel,
                        HybridLinearModel<T>& model) const {
        model.set_kernel(kernel, keys[0]);

        int i = 0;
        int D = 1;
        if (size <= 2 || num_items <= 2 || !(D <= size - 1 - D)) {
            return false;
        }

        double Ut = (model.transform(keys[size - 1 - D]) - model.transform(keys[D])) /
                    (static_cast<double>(num_items - 2)) + 1e-6;
        while (i < size - 1 - D) {
            while (i + D < size && model.transform(keys[i + D]) - model.transform(keys[i]) >= Ut) {
                i++;
            }
            if (i + D >= size) {
                break;
            }
            D = D + 1;
            if (D * 3 > size) {
                break;
            }
            if (!(D <= size - 1 - D)) {
                return false;
            }
            Ut = (model.transform(keys[size - 1 - D]) - model.transform(keys[D])) /
                 (static_cast<double>(num_items - 2)) + 1e-6;
        }

        if (D * 3 > size || !(Ut > 0)) {
            return false;
        }

        model.a = 1.0 / Ut;
        model.b = (num_items - model.a * (model.transform(keys[size - 1 - D]) +
                                          model.transform(keys[D]))) / 2;
        return isfinite(model.a) && isfinite(model.b) && model.a >= 0;
    }

    struct NormalLeafCostEstimate {
        bool valid = false;
        int num_items = 0;
        long long memory_bytes = 0;
        long double lookup_units_for_all_keys = 0;
        long double compare_units_for_all_keys = 0;
    };

    long long estimated_child_layout_bytes(int size, long long item_bytes) const {
        const int exact_items = size * static_cast<int>(compute_gap_count(size) + 1);
        const long long normal_child_bytes =
            estimated_node_bytes() +
            static_cast<long long>(exact_items) * item_bytes +
            estimated_bitmap_bytes(exact_items);

        if (size > COMPACT_LEAF_MAX_SIZE) {
            return normal_child_bytes;
        }

        // 如果父节点不 compact，冲突小子树后续仍可能被 compact。
        // 因此这里取普通布局和 compact 布局中更便宜的一个，避免夸大普通 LIPP 的成本。
        const int compact_items = compact_leaf_capacity_for_size(size);
        const long long compact_child_bytes =
            estimated_node_bytes() +
            static_cast<long long>(compact_items) * item_bytes;
        return std::min(normal_child_bytes, compact_child_bytes);
    }

    NormalLeafCostEstimate estimate_normal_leaf_cost(T* keys, int size,
                                                     long long item_bytes) const {
        NormalLeafCostEstimate result;
        const int build_gap_count = compute_gap_count(size);
        const int base_items = size * static_cast<int>(build_gap_count + 1);
        const int mid1_pos = (size - 1) / 3;
        const int mid2_pos = (size - 1) * 2 / 3;
        if (!(0 <= mid1_pos && mid1_pos < mid2_pos && mid2_pos < size - 1)) {
            return result;
        }

        const double mid1_target =
            mid1_pos * static_cast<int>(build_gap_count + 1) + static_cast<int>(build_gap_count + 1) / 2;
        const double mid2_target =
            mid2_pos * static_cast<int>(build_gap_count + 1) + static_cast<int>(build_gap_count + 1) / 2;

        HybridLinearModel<T> model;
        bool fmcd_success = false;
        if (USE_FMCD) {
            const Kernel fmcd_kernel = select_kernel_for_node(
                keys, size, base_items, mid1_pos, mid2_pos,
                mid1_target, mid2_target, true);
            fmcd_success = fit_fmcd_model(keys, size, base_items, fmcd_kernel, model);
        }
        if (!fmcd_success) {
            const Kernel fallback_kernel = select_kernel_for_node(
                keys, size, base_items, mid1_pos, mid2_pos,
                mid1_target, mid2_target, false);
            if (!fit_midpoint_model(keys, size, base_items, fallback_kernel, model)) {
                return result;
            }
        }

        int num_items = base_items;
        const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
        model.b += lr_remains;
        num_items += lr_remains * 2;

        std::vector<int> slots;
        slots.reserve(size);
        for (int i = 0; i < size; i++) {
            double v = model.predict_double(keys[i]);
            int pos;
            if (v > std::numeric_limits<int>::max() / 2) {
                pos = num_items - 1;
            } else if (v < 0) {
                pos = 0;
            } else {
                pos = std::min(num_items - 1, static_cast<int>(v));
            }
            slots.push_back(pos);
        }
        std::sort(slots.begin(), slots.end());

        long long conflict_child_bytes = 0;
        long double lookup_units_for_all_keys = static_cast<long double>(size);
        long double compare_units_for_all_keys = static_cast<long double>(size);
        for (int i = 0; i < size;) {
            int j = i + 1;
            while (j < size && slots[j] == slots[i]) {
                j++;
            }
            const int bucket = j - i;
            if (bucket > 1) {
                conflict_child_bytes += estimated_child_layout_bytes(bucket, item_bytes);
                // 父节点已经付过一次预测/访问；冲突桶里的 key 还要再进入一层小子树。
                lookup_units_for_all_keys +=
                    static_cast<long double>(bucket) *
                    std::max<long double>(1.0L, estimated_compact_lookup_units(bucket, item_bytes));
                compare_units_for_all_keys += static_cast<long double>(bucket);
            }
            i = j;
        }

        result.valid = true;
        result.num_items = num_items;
        result.memory_bytes =
            estimated_node_bytes() +
            static_cast<long long>(num_items) * item_bytes +
            estimated_bitmap_bytes(num_items) +
            conflict_child_bytes;
        result.lookup_units_for_all_keys = lookup_units_for_all_keys;
        result.compare_units_for_all_keys = compare_units_for_all_keys;
        return result;
    }

    bool should_use_local_adaptive_compact(T* keys, int size) {
        if (size < 3 || size > COMPACT_LEAF_MAX_SIZE) {
            return false;
        }
        stats.adaptive_compact_candidate_nodes++;

        const long long item_bytes = estimated_item_bytes();
        const NormalLeafCostEstimate normal = estimate_normal_leaf_cost(keys, size, item_bytes);
        if (!normal.valid) {
            return false;
        }
        const int compact_items = compact_leaf_capacity_for_size(size);
        const long long compact_memory_bytes =
            estimated_node_bytes() +
            static_cast<long long>(compact_items) * item_bytes;

        // 访问成本也折算成“会碰到多少个 Item 大小的内存块”。
        // 普通 LIPP 叶子的估算会模拟 FMCD/fallback 后的真实槽位和冲突桶；
        // compact leaf 则按实际查询实现：小叶线扫，中叶分段查找，更大才二分。
        const long double expected_lookup_ops =
            expected_operations_for_local_size(size, workload_read_ratio + workload_update_ratio +
                                                     workload_delete_ratio +
                                                     workload_scan_ratio * workload_scan_num);
        // workload 很小时，每个 key 可能还没被访问一次，不应该过度惩罚二分；
        // workload 很大时，同一片叶子会被反复访问，二分成本才需要按比例放大。
        const long double lookup_repeat_factor =
            std::max<long double>(1.0L, expected_lookup_ops / static_cast<long double>(size));
        const long long normal_access_bytes = static_cast<long long>(
            normal.lookup_units_for_all_keys * item_bytes * lookup_repeat_factor);
        const long long compact_access_bytes = static_cast<long long>(
            static_cast<long double>(size) * estimated_compact_lookup_units(size, item_bytes) *
            item_bytes * lookup_repeat_factor);

        // compact 查询即使 cache 友好，也要多做 key 比较和分支判断。
        // 这里把 CPU 比较成本折算成固定字节成本，避免模型只看“省内存”而忽略吞吐下降。
        const long long normal_compare_bytes = static_cast<long long>(
            normal.compare_units_for_all_keys *
            static_cast<long double>(COMPACT_COMPARE_COST_BYTES) *
            lookup_repeat_factor);
        const long long compact_compare_bytes = static_cast<long long>(
            static_cast<long double>(size) * estimated_compact_compare_units(size) *
            static_cast<long double>(COMPACT_COMPARE_COST_BYTES) *
            lookup_repeat_factor);

        // compact 插入需要搬移连续数组。这里按当前 workload 估算会插入到这个小块的次数；
        // 读多时主要惩罚二分查找，写多时主要惩罚数组搬移。
        const long double expected_insert_ops =
            expected_operations_for_local_size(size, workload_insert_ratio + workload_delete_ratio);
        const long long normal_insert_bytes = static_cast<long long>(
            expected_insert_ops * item_bytes);
        const long long compact_insert_bytes = static_cast<long long>(
            expected_insert_ops * (static_cast<long double>(size) * item_bytes / 2.0L));

        const long long normal_total_cost =
            normal.memory_bytes + normal_access_bytes + normal_compare_bytes + normal_insert_bytes;
        const long long compact_total_cost =
            compact_memory_bytes + compact_access_bytes + compact_compare_bytes + compact_insert_bytes +
            // 切换成 compact leaf 不是免费的：查询要走另一套路径，插入要搬移数组。
            // 用 2 条 cache line 表示这类固定管理成本，避免为了几十字节的小收益改变布局。
            128;

        const bool choose_compact = compact_total_cost < normal_total_cost;
        if (choose_compact) {
            stats.adaptive_compact_estimated_saved_bytes +=
                std::max<long long>(0, normal.memory_bytes - compact_memory_bytes);
            stats.adaptive_compact_estimated_extra_access_bytes +=
                std::max<long long>(0, (compact_access_bytes + compact_compare_bytes) -
                                           (normal_access_bytes + normal_compare_bytes));
        }
        return choose_compact;
    }

    bool should_use_compact_leaf(T* keys, int size) {
        if (size > COMPACT_LEAF_MAX_SIZE) {
            return false;
        }
        if (compact_policy == LIPPHybridCompactPolicy::FORCE) {
            return true;
        }
        if (compact_policy == LIPPHybridCompactPolicy::ADAPTIVE) {
            const bool use_compact = should_use_local_adaptive_compact(keys, size);
            if (use_compact) {
                adaptive_compact_enabled = true;
            }
            return use_compact;
        }
        return false;
    }

    Kernel select_kernel_for_node(T* keys, int size, int num_items,
                                  int mid1_pos, int mid2_pos,
                                  double mid1_target, double mid2_target,
                                  bool fmcd_main_path) const {
        if (kernel_policy == LIPPHybridKernelPolicy::IDENTITY_ONLY) {
            return HybridLinearModel<T>::IDENTITY;
        }
        if (kernel_policy == LIPPHybridKernelPolicy::FORCE_LOG1P ||
            kernel_policy == LIPPHybridKernelPolicy::FORCE_SQRT ||
            kernel_policy == LIPPHybridKernelPolicy::FORCE_CBRT) {
            return forced_kernel();
        }
        if (kernel_policy == LIPPHybridKernelPolicy::FALLBACK_ONLY && fmcd_main_path) {
            return HybridLinearModel<T>::IDENTITY;
        }
        if (kernel_policy == LIPPHybridKernelPolicy::ADAPTIVE_AGGRESSIVE && sparse_key_space_sqrt) {
            // 根节点判断出整棵树属于稀疏大跨度 key 空间后，子树统一沿用 sqrt。
            // 这是一个粗粒度自适应开关，用来验证“按数据集分布选择核函数”是否有收益。
            return HybridLinearModel<T>::SQRT;
        }
        if (kernel_policy == LIPPHybridKernelPolicy::ADAPTIVE_AGGRESSIVE) {
            // 当前阶段先验证“全局 hardness 触发 sqrt”这一条线。
            // 如果全局条件不成立，不再让局部抽样估计器零散选择 sqrt，避免 OSM 5M 这类场景误触发。
            return HybridLinearModel<T>::IDENTITY;
        }

        // 小节点本身冲突成本很低，保持 identity 可以减少预测开销。
        if (size < 64 || keys[0] == keys[size - 1]) {
            return HybridLinearModel<T>::IDENTITY;
        }

        Kernel candidates[] = {
            HybridLinearModel<T>::IDENTITY,
            HybridLinearModel<T>::SQRT
        };

        const long long identity_cost = fmcd_main_path
            ? estimate_fmcd_kernel_cost(keys, size, num_items, HybridLinearModel<T>::IDENTITY)
            : estimate_midpoint_kernel_cost(keys, size, num_items, HybridLinearModel<T>::IDENTITY,
                                            mid1_pos, mid2_pos, mid1_target, mid2_target);

        Kernel best_kernel = HybridLinearModel<T>::IDENTITY;
        long long best_cost = identity_cost;
        for (Kernel kernel : candidates) {
            if (kernel == HybridLinearModel<T>::IDENTITY) {
                continue;
            }
            const long long cost = fmcd_main_path
                ? estimate_fmcd_kernel_cost(keys, size, num_items, kernel)
                : estimate_midpoint_kernel_cost(keys, size, num_items, kernel,
                                                mid1_pos, mid2_pos, mid1_target, mid2_target);
            if (cost < best_cost) {
                best_cost = cost;
                best_kernel = kernel;
            }
        }

        // 保守策略要求冲突成本至少下降 25%；激进策略只要求比 identity 略好。
        // 这样可以单独验证“核函数是否有用”，而不把策略门槛混进结论里。
        if (best_kernel == HybridLinearModel<T>::IDENTITY) {
            return HybridLinearModel<T>::IDENTITY;
        }
        if (kernel_policy == LIPPHybridKernelPolicy::ADAPTIVE_AGGRESSIVE && best_cost < identity_cost) {
            return best_kernel;
        }
        if (best_cost * 4 < identity_cost * 3) {
            return best_kernel;
        }
        return HybridLinearModel<T>::IDENTITY;
    }

    void record_selected_kernel(Kernel kernel) {
        if (kernel == HybridLinearModel<T>::IDENTITY) {
            return;
        }
        stats.adaptive_kernel_nodes++;
        if (kernel == HybridLinearModel<T>::LOG1P) {
            stats.adaptive_kernel_log1p_nodes++;
        } else if (kernel == HybridLinearModel<T>::SQRT) {
            stats.adaptive_kernel_sqrt_nodes++;
        } else if (kernel == HybridLinearModel<T>::CBRT) {
            stats.adaptive_kernel_cbrt_nodes++;
        }
    }

    const double BUILD_LR_REMAIN;
    const bool QUIET;
    const LIPPHybridKernelPolicy kernel_policy;
    const LIPPHybridCompactPolicy compact_policy;
    bool sparse_key_space_sqrt = false;
    bool adaptive_compact_enabled = false;
    long double workload_read_ratio = 1.0L;
    long double workload_insert_ratio = 0.0L;
    long double workload_update_ratio = 0.0L;
    long double workload_scan_ratio = 0.0L;
    long double workload_delete_ratio = 0.0L;
    long double workload_scan_num = 100.0L;
    long long workload_operations_num = 0;
    long long workload_total_keys = 0;
    static constexpr int COMPACT_LEAF_MAX_SIZE = 64;
    static constexpr int COMPACT_LINEAR_SCAN_MAX_SIZE = 16;
    static constexpr int COMPACT_SEGMENT_SCAN_MAX_SIZE = 64;
    static constexpr int COMPACT_SEGMENT_SIZE = 8;
    static constexpr long long COMPACT_COMPARE_COST_BYTES = 16;

    struct {
        long long fmcd_success_times = 0;
        long long fmcd_broken_times = 0;
        long long adaptive_kernel_nodes = 0;
        long long adaptive_kernel_log1p_nodes = 0;
        long long adaptive_kernel_sqrt_nodes = 0;
        long long adaptive_kernel_cbrt_nodes = 0;
        long long adaptive_compact_candidate_nodes = 0;
        long long adaptive_compact_estimated_saved_bytes = 0;
        long long adaptive_compact_estimated_extra_access_bytes = 0;
        long long compact_leaf_nodes = 0;
        #if COLLECT_TIME
        double time_scan_and_destory_tree = 0;
        double time_build_tree_bulk = 0;
        #endif
    } stats;

public:
    typedef std::pair<T, P> V;

    LIPPHybrid(double BUILD_LR_REMAIN = 0, bool QUIET = true,
               LIPPHybridKernelPolicy kernel_policy = LIPPHybridKernelPolicy::FALLBACK_ONLY,
               LIPPHybridCompactPolicy compact_policy = LIPPHybridCompactPolicy::DISABLED)
        : BUILD_LR_REMAIN(BUILD_LR_REMAIN),
          QUIET(QUIET),
          kernel_policy(kernel_policy),
          compact_policy(compact_policy) {
        {
            std::vector<Node*> nodes;
            for (int _ = 0; _ < 1e7; _ ++) {
                Node* node = build_tree_two(T(0), P(), T(1), P());
                nodes.push_back(node);
            }
            for (auto node : nodes) {
                destroy_tree(node);
            }
            if (!QUIET) {
                printf("initial memory pool size = %lu\n", pending_two.size());
            }
        }
        if (USE_FMCD && !QUIET) {
            printf("enable FMCD\n");
        }

        root = build_tree_none();
    }
    ~LIPPHybrid() {
        destroy_tree(root);
        root = NULL;
        destory_pending();
    }

    void set_workload(double read_ratio, double insert_ratio, double update_ratio,
                      double scan_ratio, double delete_ratio, long long operations_num,
                      long long scan_num) {
        // 这些信息来自 benchmark，用来判断 compact leaf 的 CPU 代价会被重复支付多少次。
        // 没有传入时保持默认值，索引仍可独立使用。
        workload_read_ratio = read_ratio;
        workload_insert_ratio = insert_ratio;
        workload_update_ratio = update_ratio;
        workload_scan_ratio = scan_ratio;
        workload_delete_ratio = delete_ratio;
        workload_operations_num = operations_num;
        workload_scan_num = scan_num;
    }

    bool insert(const V& v) {
        return insert(v.first, v.second);
    }
    bool insert(const T& key, const P& value) {
        bool ok = true;
        root = insert_tree(root, key, value, &ok);
        return ok;
    }
    P at(const T& key, bool skip_existence_check, bool& exist) const {
        Node* node = root;
        exist = true;

        while (true) {
            if (node->is_compact) {
                const int pos = compact_lower_bound(node, key);
                if (pos < node->size && node->items[pos].comp.data.key == key) {
                    return node->items[pos].comp.data.value;
                }
                exist = false;
                return static_cast<P>(0);
            }
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->child_bitmap, pos) == 1) {
                node = node->items[pos].comp.child;
            } else {
                if (skip_existence_check) {
                    return node->items[pos].comp.data.value;
                } else {
                    if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                        exist = false;
                        return static_cast<P>(0);
                    } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                        LIPPHYBRID_RT_ASSERT(node->items[pos].comp.data.key == key);
                        return node->items[pos].comp.data.value;
                    }
                }
            }
        }
    }
    bool exists(const T& key) const {
        Node* node = root;
        while (true) {
            if (node->is_compact) {
                const int pos = compact_lower_bound(node, key);
                return pos < node->size && node->items[pos].comp.data.key == key;
            }
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                return false;
            } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                return node->items[pos].comp.data.key == key;
            } else {
                node = node->items[pos].comp.child;
            }
        }
    }
    void bulk_load(const V* vs, int num_keys) {
        sparse_key_space_sqrt = false;
        adaptive_compact_enabled = false;
        if (num_keys == 0) {
            destroy_tree(root);
            root = build_tree_none();
            return;
        }
        if (num_keys == 1) {
            destroy_tree(root);
            root = build_tree_none();
            insert(vs[0]);
            return;
        }
        if (num_keys == 2) {
            destroy_tree(root);
            root = build_tree_two(vs[0].first, vs[0].second, vs[1].first, vs[1].second);
            return;
        }

        LIPPHYBRID_RT_ASSERT(num_keys > 2);
        workload_total_keys = num_keys;
        for (int i = 1; i < num_keys; i ++) {
            LIPPHYBRID_RT_ASSERT(vs[i].first > vs[i-1].first);
        }

        T* keys = new T[num_keys];
        P* values = new P[num_keys];
        for (int i = 0; i < num_keys; i ++) {
            keys[i] = vs[i].first;
            values[i] = vs[i].second;
        }
        // sqrt 仍然用全局 hardness；compact leaf 已改为建树过程中的子树级判断。
        // adaptive_compact_enabled 不再是预先打开的全局开关，而表示是否至少有一个小块被选中。
        sparse_key_space_sqrt = should_enable_global_sqrt(keys, num_keys);
        destroy_tree(root);
        root = build_tree_bulk(keys, values, num_keys);
        delete[] keys;
        delete[] values;
    }

    // 下面几个统计接口先服务于 lipphybrid 的消融实验。
    // 当前 benchmark 还没有单独的 LIPP 统计列，所以包装层会临时映射到通用计数字段。
    long long adaptive_kernel_nodes() const { return stats.adaptive_kernel_nodes; }
    long long adaptive_kernel_log1p_nodes() const { return stats.adaptive_kernel_log1p_nodes; }
    long long adaptive_kernel_sqrt_nodes() const { return stats.adaptive_kernel_sqrt_nodes; }
    long long adaptive_kernel_cbrt_nodes() const { return stats.adaptive_kernel_cbrt_nodes; }
    long long hardness_sqrt_enabled() const { return sparse_key_space_sqrt ? 1 : 0; }
    long long adaptive_compact_enabled_stat() const { return adaptive_compact_enabled ? 1 : 0; }
    long long adaptive_compact_candidate_nodes() const { return stats.adaptive_compact_candidate_nodes; }
    long long adaptive_compact_estimated_saved_bytes() const { return stats.adaptive_compact_estimated_saved_bytes; }
    long long adaptive_compact_estimated_extra_access_bytes() const { return stats.adaptive_compact_estimated_extra_access_bytes; }
    long long compact_leaf_nodes() const { return stats.compact_leaf_nodes; }

    bool remove(const T &key) {
        constexpr int MAX_DEPTH = 128;
        Node *path[MAX_DEPTH];
        int path_size = 0;
        Node *parent = nullptr;

        for (Node* node = root; ; ) {
            LIPPHYBRID_RT_ASSERT(path_size < MAX_DEPTH);
            path[path_size++] = node;
            if (node->is_compact) {
                const int pos = compact_lower_bound(node, key);
                if (pos >= node->size || node->items[pos].comp.data.key != key) {
                    return false;
                }
                for (int i = pos + 1; i < node->size; i++) {
                    node->items[i - 1] = node->items[i];
                }
                for (int i = 0; i < path_size; i++) {
                    path[i]->size--;
                }
                return true;
            }
            // node->size--;
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->child_bitmap, pos) == 1) {
                parent = node;
                node = node->items[pos].comp.child;
            } else if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                return false;
            } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                BITMAP_SET(node->none_bitmap, pos);
                for(int i = 0; i < path_size; i++) {
                    path[i]->size--;
                }
                if(node->size == 0) {
                    int parent_pos = PREDICT_POS(parent, key);
                    BITMAP_CLEAR(parent->child_bitmap, parent_pos);
                    BITMAP_SET(parent->none_bitmap, parent_pos);
                    delete_items(node->items, node->num_items);
                    const int bitmap_size = BITMAP_SIZE(node->num_items);
                    delete_bitmap(node->none_bitmap, bitmap_size);
                    delete_bitmap(node->child_bitmap, bitmap_size);
                    delete_nodes(node, 1);
                }
                return true;
            }
        }
    }

    bool update(const T &key, const P& value) {
        for (Node* node = root; ; ) {
            if (node->is_compact) {
                const int pos = compact_lower_bound(node, key);
                if (pos < node->size && node->items[pos].comp.data.key == key) {
                    node->items[pos].comp.data.value = value;
                    return true;
                }
                return false;
            }
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                return false;
            } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                node->items[pos].comp.data.value = value;
                return true;
            } else {
                node = node->items[pos].comp.child;
            }
        }
    }

    // Find the minimum `len` keys which are no less than `lower`, returns the number of found keys.
    int range_query_len(std::pair<T,P>* results, const T& lower, int len) {
        return range_core_len<false>(results, 0, root, lower, len);
    }

    void show() const {
        printf("============= SHOW LIPPHYBRID ================\n");

        std::stack<Node*> s;
        s.push(root);
        while (!s.empty()) {
            Node* node = s.top(); s.pop();

            printf("Node(%p, a = %lf, b = %lf, num_items = %d)", node, node->model.a, node->model.b, node->num_items);
            printf("[");
            if (node->is_compact) {
                for (int i = 0; i < node->size; i++) {
                    if (i) {
                        printf(", ");
                    }
                    std::stringstream ss;
                    ss << node->items[i].comp.data.key;
                    printf("Key(%s)", ss.str().c_str());
                }
                printf("]\n");
                continue;
            }
            int first = 1;
            for (int i = 0; i < node->num_items; i ++) {
                if (!first) {
                    printf(", ");
                }
                first = 0;
                if (BITMAP_GET(node->none_bitmap, i) == 1) {
                    printf("None");
                } else if (BITMAP_GET(node->child_bitmap, i) == 0) {
                    std::stringstream s;
                    s << node->items[i].comp.data.key;
                    printf("Key(%s)", s.str().c_str());
                } else {
                    printf("Child(%p)", node->items[i].comp.child);
                    s.push(node->items[i].comp.child);
                }
            }
            printf("]\n");
        }
    }
    void print_depth() const {
        std::stack<Node*> s;
        std::stack<int> d;
        s.push(root);
        d.push(1);

        int max_depth = 1;
        int sum_depth = 0, sum_nodes = 0;
        while (!s.empty()) {
            Node* node = s.top(); s.pop();
            int depth = d.top(); d.pop();
            if (node->is_compact) {
                max_depth = std::max(max_depth, depth);
                sum_depth += depth * node->size;
                sum_nodes += node->size;
                continue;
            }
            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                    d.push(depth + 1);
                } else if (BITMAP_GET(node->none_bitmap, i) != 1) {
                    max_depth = std::max(max_depth, depth);
                    sum_depth += depth;
                    sum_nodes ++;
                }
            }
        }

        printf("max_depth = %d, avg_depth = %.2lf\n", max_depth, double(sum_depth) / double(sum_nodes));
    }
    void verify() const {
        std::stack<Node*> s;
        s.push(root);

        while (!s.empty()) {
            Node* node = s.top(); s.pop();
            if (node->is_compact) {
                LIPPHYBRID_RT_ASSERT(node->size <= node->num_items);
                for (int i = 1; i < node->size; i++) {
                    LIPPHYBRID_RT_ASSERT(node->items[i - 1].comp.data.key < node->items[i].comp.data.key);
                }
                continue;
            }
            int sum_size = 0;
            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                    sum_size += node->items[i].comp.child->size;
                } else if (BITMAP_GET(node->none_bitmap, i) != 1) {
                    sum_size ++;
                }
            }
            LIPPHYBRID_RT_ASSERT(sum_size == node->size);
        }
    }
    void print_stats() const {
        printf("======== Stats ===========\n");
        if (USE_FMCD) {
            printf("\t fmcd_success_times = %lld\n", stats.fmcd_success_times);
            printf("\t fmcd_broken_times = %lld\n", stats.fmcd_broken_times);
        }
        #if COLLECT_TIME
        printf("\t time_scan_and_destory_tree = %lf\n", stats.time_scan_and_destory_tree);
        printf("\t time_build_tree_bulk = %lf\n", stats.time_build_tree_bulk);
        #endif
    }
    size_t index_size() const {
        std::stack<Node*> s;
        s.push(root);

        size_t size = 0;
        while (!s.empty()) {
            Node* node = s.top(); s.pop();
            size += sizeof(*node);
            if (node->is_compact) {
                size += sizeof(Item) * node->num_items;
                continue;
            }
            size += sizeof(*(node->none_bitmap));
            size += sizeof(*(node->child_bitmap));
            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                    size += sizeof(Item);
                } else {
                    if (BITMAP_GET(node->none_bitmap, i) == 1) {
                        size += sizeof(Item);
                    } 
                }
            }
        }
        return size;
    }
    size_t total_size() const {
        std::stack < Node * > s;
        s.push(root);

        size_t size = 0;
        while (!s.empty()) {
            Node *node = s.top();
            s.pop();
            size += sizeof(*node);
            if (node->is_compact) {
                size += sizeof(Item) * node->num_items;
                continue;
            }
            size += sizeof(*(node->none_bitmap));
            size += sizeof(*(node->child_bitmap));
            for (int i = 0; i < node->num_items; i++) {
                size += sizeof(Item);
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                }
            }
        }
        return size;
    }

private:
    struct Node;
    struct Item
    {
        union {
            struct {
                T key;
                P value;
            } data;
            Node* child;
        } comp;
    };
    struct Node
    {
        int is_two; // is special node for only two keys
        int is_compact; // 紧凑叶节点：items[0..size) 连续存放有序 key，不再使用精确槽位。
        int build_size; // tree size (include sub nodes) when node created
        int size; // current tree size (include sub nodes)
        int fixed; // fixed node will not trigger rebuild
        int num_inserts, num_insert_to_data;
        int num_items; // size of items
        HybridLinearModel<T> model;
        Item* items;
        bitmap_t* none_bitmap; // 1 means None, 0 means Data or Child
        bitmap_t* child_bitmap; // 1 means Child. will always be 0 when none_bitmap is 1
    };

    Node* root;
    std::stack<Node*> pending_two;

    std::allocator<Node> node_allocator;
    Node* new_nodes(int n)
    {
        Node* p = node_allocator.allocate(n);
        LIPPHYBRID_RT_ASSERT(p != NULL && p != (Node*)(-1));
        return p;
    }
    void delete_nodes(Node* p, int n)
    {
        node_allocator.deallocate(p, n);
    }

    std::allocator<Item> item_allocator;
    Item* new_items(int n)
    {
        Item* p = item_allocator.allocate(n);
        LIPPHYBRID_RT_ASSERT(p != NULL && p != (Item*)(-1));
        return p;
    }
    void delete_items(Item* p, int n)
    {
        item_allocator.deallocate(p, n);
    }

    std::allocator<bitmap_t> bitmap_allocator;
    bitmap_t* new_bitmap(int n)
    {
        bitmap_t* p = bitmap_allocator.allocate(n);
        LIPPHYBRID_RT_ASSERT(p != NULL && p != (bitmap_t*)(-1));
        return p;
    }
    void delete_bitmap(bitmap_t* p, int n)
    {
        bitmap_allocator.deallocate(p, n);
    }

    /// build an empty tree
    Node* build_tree_none()
    {
        Node* node = new_nodes(1);
        node->is_two = 0;
        node->is_compact = 0;
        node->build_size = 0;
        node->size = 0;
        node->fixed = 0;
        node->num_inserts = node->num_insert_to_data = 0;
        node->num_items = 1;
        node->model.a = node->model.b = 0;
        node->items = new_items(1);
        node->none_bitmap = new_bitmap(1);
        node->none_bitmap[0] = 0;
        BITMAP_SET(node->none_bitmap, 0);
        node->child_bitmap = new_bitmap(1);
        node->child_bitmap[0] = 0;

        return node;
    }
    /// build a tree with two keys
    Node* build_tree_two(T key1, P value1, T key2, P value2)
    {
        if (key1 > key2) {
            std::swap(key1, key2);
            std::swap(value1, value2);
        }
        LIPPHYBRID_RT_ASSERT(key1 < key2);
        static_assert(BITMAP_WIDTH == 8);

        Node* node = NULL;
        if (pending_two.empty()) {
            node = new_nodes(1);
            node->is_two = 1;
            node->is_compact = 0;
            node->build_size = 2;
            node->size = 2;
            node->fixed = 0;
            node->num_inserts = node->num_insert_to_data = 0;

            node->num_items = 8;
            node->items = new_items(node->num_items);
            node->none_bitmap = new_bitmap(1);
            node->child_bitmap = new_bitmap(1);
            node->none_bitmap[0] = 0xff;
            node->child_bitmap[0] = 0;
        } else {
            node = pending_two.top(); pending_two.pop();
            node->is_compact = 0;
        }

        const long double mid1_key = key1;
        const long double mid2_key = key2;
        node->model.set_kernel(HybridLinearModel<T>::IDENTITY, key1);

        const double mid1_target = node->num_items / 3;
        const double mid2_target = node->num_items * 2 / 3;

        const long double tx1 = node->model.transform_value(mid1_key);
        const long double tx2 = node->model.transform_value(mid2_key);
        node->model.a = (mid2_target - mid1_target) / (tx2 - tx1);
        node->model.b = mid1_target - node->model.a * tx1;
        LIPPHYBRID_RT_ASSERT(isfinite(node->model.a));
        LIPPHYBRID_RT_ASSERT(isfinite(node->model.b));

        { // insert key1&value1
            int pos = PREDICT_POS(node, key1);
            LIPPHYBRID_RT_ASSERT(BITMAP_GET(node->none_bitmap, pos) == 1);
            BITMAP_CLEAR(node->none_bitmap, pos);
            node->items[pos].comp.data.key = key1;
            node->items[pos].comp.data.value = value1;
        }
        { // insert key2&value2
            int pos = PREDICT_POS(node, key2);
            LIPPHYBRID_RT_ASSERT(BITMAP_GET(node->none_bitmap, pos) == 1);
            BITMAP_CLEAR(node->none_bitmap, pos);
            node->items[pos].comp.data.key = key2;
            node->items[pos].comp.data.value = value2;
        }

        return node;
    }

    int compact_lower_bound(Node* node, const T& key) const
    {
        return compact_lower_bound(node, key, node->size);
    }

    int compact_lower_bound(Node* node, const T& key, int valid_size) const
    {
        if (valid_size <= COMPACT_LINEAR_SCAN_MAX_SIZE) {
            // 小 compact leaf 连续存放在数组里。此时线性扫描常常比二分更快：
            // 1. 访问顺序连续，cache 友好；
            // 2. 分支形态简单，避免二分查找每一步跳到不同位置。
            for (int i = 0; i < valid_size; i++) {
                if (!(node->items[i].comp.data.key < key)) {
                    return i;
                }
            }
            return valid_size;
        }

        if (valid_size <= COMPACT_SEGMENT_SCAN_MAX_SIZE) {
            // 中等大小的 compact leaf 不直接二分。
            // 做法：每 8 个元素看一次段尾 key，先找到可能所在的小段，再在段内线性扫。
            // 这样既避免全量线扫，也比二分更顺序、更 cache 友好。
            int block_begin = 0;
            while (block_begin < valid_size) {
                const int block_end = std::min(valid_size, block_begin + COMPACT_SEGMENT_SIZE);
                if (!(node->items[block_end - 1].comp.data.key < key)) {
                    for (int i = block_begin; i < block_end; i++) {
                        if (!(node->items[i].comp.data.key < key)) {
                            return i;
                        }
                    }
                    return block_end;
                }
                block_begin = block_end;
            }
            return valid_size;
        }

        int l = 0;
        int r = valid_size;
        while (l < r) {
            const int m = l + (r - l) / 2;
            if (node->items[m].comp.data.key < key) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        return l;
    }

    void init_compact_leaf(Node* node, T* keys, P* values, int size)
    {
        node->is_two = 0;
        node->is_compact = 1;
        node->build_size = size;
        node->size = size;
        node->fixed = 0;
        node->num_inserts = node->num_insert_to_data = 0;
        // 紧凑叶节点预留少量连续空间；插入时用数组搬移，不再为每个 key 保留精确槽位。
        node->num_items = compact_leaf_capacity_for_size(size);
        node->model.set_kernel(HybridLinearModel<T>::IDENTITY, keys[0]);
        node->model.a = node->model.b = 0;
        node->items = new_items(node->num_items);
        node->none_bitmap = nullptr;
        node->child_bitmap = nullptr;
        for (int i = 0; i < size; i++) {
            node->items[i].comp.data.key = keys[i];
            node->items[i].comp.data.value = values[i];
        }
        stats.compact_leaf_nodes++;
    }

    Node* build_tree_compact(T* keys, P* values, int size)
    {
        Node* node = new_nodes(1);
        init_compact_leaf(node, keys, values, size);
        return node;
    }

    /// bulk build, _keys must be sorted in asc order.
    Node* build_tree_bulk(T* _keys, P* _values, int _size)
    {
        if (USE_FMCD) {
            return build_tree_bulk_fmcd(_keys, _values, _size);
        } else {
            return build_tree_bulk_fast(_keys, _values, _size);
        }
    }
    /// bulk build, _keys must be sorted in asc order.
    /// split keys into three parts at each node.
    Node* build_tree_bulk_fast(T* _keys, P* _values, int _size)
    {
        LIPPHYBRID_RT_ASSERT(_size > 1);

        typedef struct {
            int begin;
            int end;
            int level; // top level = 1
            Node* node;
        } Segment;
        std::stack<Segment> s;

        Node* ret = new_nodes(1);
        s.push((Segment){0, _size, 1, ret});

        while (!s.empty()) {
            const int begin = s.top().begin;
            const int end = s.top().end;
            const int level = s.top().level;
            Node* node = s.top().node;
            s.pop();

            LIPPHYBRID_RT_ASSERT(end - begin >= 2);
            if (should_use_compact_leaf(_keys + begin, end - begin)) {
                init_compact_leaf(node, _keys + begin, _values + begin, end - begin);
            } else if (end - begin == 2) {
                Node* _ = build_tree_two(_keys[begin], _values[begin], _keys[begin+1], _values[begin+1]);
                memcpy(node, _, sizeof(Node));
                delete_nodes(_, 1);
            } else {
                T* keys = _keys + begin;
                P* values = _values + begin;
                const int size = end - begin;
                const int BUILD_GAP_CNT = compute_gap_count(size);

                node->is_two = 0;
                node->is_compact = 0;
                node->build_size = size;
                node->size = size;
                node->fixed = 0;
                node->num_inserts = node->num_insert_to_data = 0;

                int mid1_pos = (size - 1) / 3;
                int mid2_pos = (size - 1) * 2 / 3;

                LIPPHYBRID_RT_ASSERT(0 <= mid1_pos);
                LIPPHYBRID_RT_ASSERT(mid1_pos < mid2_pos);
                LIPPHYBRID_RT_ASSERT(mid2_pos < size - 1);

                const long double mid1_key =
                        (static_cast<long double>(keys[mid1_pos]) + static_cast<long double>(keys[mid1_pos + 1])) / 2;
                const long double mid2_key =
                        (static_cast<long double>(keys[mid2_pos]) + static_cast<long double>(keys[mid2_pos + 1])) / 2;

	                node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
	                const double mid1_target = mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
	                const double mid2_target = mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
                    const Kernel selected_kernel = select_kernel_for_node(
                        keys, size, node->num_items, mid1_pos, mid2_pos, mid1_target, mid2_target, false);
                    node->model.set_kernel(selected_kernel, keys[0]);
                    record_selected_kernel(selected_kernel);

                    const long double tx1 = node->model.transform_value(mid1_key);
                    const long double tx2 = node->model.transform_value(mid2_key);
	                node->model.a = (mid2_target - mid1_target) / (tx2 - tx1);
	                node->model.b = mid1_target - node->model.a * tx1;
                LIPPHYBRID_RT_ASSERT(isfinite(node->model.a));
                LIPPHYBRID_RT_ASSERT(isfinite(node->model.b));

                const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
                node->model.b += lr_remains;
                node->num_items += lr_remains * 2;

                if (size > 1e6) {
                    node->fixed = 1;
                }

                node->items = new_items(node->num_items);
                const int bitmap_size = BITMAP_SIZE(node->num_items);
                node->none_bitmap = new_bitmap(bitmap_size);
                node->child_bitmap = new_bitmap(bitmap_size);
                memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
                memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);

                for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size; ) {
                    int next = offset + 1, next_i = -1;
                    while (next < size) {
                        next_i = PREDICT_POS(node, keys[next]);
                        if (next_i == item_i) {
                            next ++;
                        } else {
                            break;
                        }
                    }
                    if (next == offset + 1) {
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        node->items[item_i].comp.data.key = keys[offset];
                        node->items[item_i].comp.data.value = values[offset];
                    } else {
                        // ASSERT(next - offset <= (size+2) / 3);
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        BITMAP_SET(node->child_bitmap, item_i);
                        node->items[item_i].comp.child = new_nodes(1);
                        s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});
                    }
                    if (next >= size) {
                        break;
                    } else {
                        item_i = next_i;
                        offset = next;
                    }
                }
            }
        }

        return ret;
    }
    /// bulk build, _keys must be sorted in asc order.
    /// FMCD method.
    Node* build_tree_bulk_fmcd(T* _keys, P* _values, int _size)
    {
        LIPPHYBRID_RT_ASSERT(_size > 1);

        typedef struct {
            int begin;
            int end;
            int level; // top level = 1
            Node* node;
        } Segment;
        std::stack<Segment> s;

        Node* ret = new_nodes(1);
        s.push((Segment){0, _size, 1, ret});

        while (!s.empty()) {
            const int begin = s.top().begin;
            const int end = s.top().end;
            const int level = s.top().level;
            Node* node = s.top().node;
            s.pop();

            LIPPHYBRID_RT_ASSERT(end - begin >= 2);
            if (should_use_compact_leaf(_keys + begin, end - begin)) {
                init_compact_leaf(node, _keys + begin, _values + begin, end - begin);
            } else if (end - begin == 2) {
                Node* _ = build_tree_two(_keys[begin], _values[begin], _keys[begin+1], _values[begin+1]);
                memcpy(node, _, sizeof(Node));
                delete_nodes(_, 1);
            } else {
                T* keys = _keys + begin;
                P* values = _values + begin;
                const int size = end - begin;
                const int BUILD_GAP_CNT = compute_gap_count(size);

                node->is_two = 0;
                node->is_compact = 0;
                node->build_size = size;
                node->size = size;
                node->fixed = 0;
                node->num_inserts = node->num_insert_to_data = 0;

                // FMCD method
                // Here the implementation is a little different with Algorithm 1 in our paper.
                // In Algorithm 1, U_T should be (keys[size-1-D] - keys[D]) / (L - 2).
                // But according to the derivation described in our paper, M.A should be less than 1 / U_T.
                // So we added a small number (1e-6) to U_T.
                // In fact, it has only a negligible impact of the performance.
	                {
	                    const int L = size * static_cast<int>(BUILD_GAP_CNT + 1);
                        // FMCD 本身就是 LIPP 为降低冲突设计的主方法。
                        // 默认策略保持 FMCD 主路径为 identity；激进/强制策略会在这里验证非线性核。
                        const int kernel_mid1_pos = (size - 1) / 3;
                        const int kernel_mid2_pos = (size - 1) * 2 / 3;
                        const double kernel_mid1_target =
                            kernel_mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) +
                            static_cast<int>(BUILD_GAP_CNT + 1) / 2;
                        const double kernel_mid2_target =
                            kernel_mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) +
                            static_cast<int>(BUILD_GAP_CNT + 1) / 2;
                        const Kernel selected_kernel = select_kernel_for_node(
                            keys, size, L, kernel_mid1_pos, kernel_mid2_pos,
                            kernel_mid1_target, kernel_mid2_target, true);
                        node->model.set_kernel(selected_kernel, keys[0]);
                        record_selected_kernel(selected_kernel);
	                    int i = 0;
	                    int D = 1;
	                    LIPPHYBRID_RT_ASSERT(D <= size-1-D);
	                    double Ut = (node->model.transform(keys[size - 1 - D]) - node->model.transform(keys[D])) /
	                                (static_cast<double>(L - 2)) + 1e-6;
	                    while (i < size - 1 - D) {
	                        while (i + D < size && node->model.transform(keys[i + D]) - node->model.transform(keys[i]) >= Ut) {
	                            i ++;
	                        }
                        if (i + D >= size) {
                            break;
                        }
                        D = D + 1;
	                        if (D * 3 > size) break;
	                        LIPPHYBRID_RT_ASSERT(D <= size-1-D);
	                        Ut = (node->model.transform(keys[size - 1 - D]) - node->model.transform(keys[D])) /
	                             (static_cast<double>(L - 2)) + 1e-6;
	                    }
                    if (D * 3 <= size) {
                        stats.fmcd_success_times ++;

	                        node->model.a = 1.0 / Ut;
	                        node->model.b = (L - node->model.a * (node->model.transform(keys[size - 1 - D]) +
	                                                              node->model.transform(keys[D]))) / 2;
                        LIPPHYBRID_RT_ASSERT(isfinite(node->model.a));
                        LIPPHYBRID_RT_ASSERT(isfinite(node->model.b));
                        node->num_items = L;
                    } else {
                        stats.fmcd_broken_times ++;

                        int mid1_pos = (size - 1) / 3;
                        int mid2_pos = (size - 1) * 2 / 3;

                        LIPPHYBRID_RT_ASSERT(0 <= mid1_pos);
                        LIPPHYBRID_RT_ASSERT(mid1_pos < mid2_pos);
                        LIPPHYBRID_RT_ASSERT(mid2_pos < size - 1);

                        const long double mid1_key = (static_cast<long double>(keys[mid1_pos]) +
                                                      static_cast<long double>(keys[mid1_pos + 1])) / 2;
                        const long double mid2_key = (static_cast<long double>(keys[mid2_pos]) +
                                                      static_cast<long double>(keys[mid2_pos + 1])) / 2;

                        node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
	                        const double mid1_target = mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
	                        const double mid2_target = mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
                            const Kernel selected_kernel = select_kernel_for_node(
                                keys, size, node->num_items, mid1_pos, mid2_pos, mid1_target, mid2_target, false);
                            node->model.set_kernel(selected_kernel, keys[0]);
                            record_selected_kernel(selected_kernel);

                            const long double tx1 = node->model.transform_value(mid1_key);
                            const long double tx2 = node->model.transform_value(mid2_key);
	                        node->model.a = (mid2_target - mid1_target) / (tx2 - tx1);
	                        node->model.b = mid1_target - node->model.a * tx1;
                        LIPPHYBRID_RT_ASSERT(isfinite(node->model.a));
                        LIPPHYBRID_RT_ASSERT(isfinite(node->model.b));
                    }
                }
                LIPPHYBRID_RT_ASSERT(node->model.a >= 0);
                const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
                node->model.b += lr_remains;
                node->num_items += lr_remains * 2;

                if (size > 1e6) {
                    node->fixed = 1;
                }

                node->items = new_items(node->num_items);
                const int bitmap_size = BITMAP_SIZE(node->num_items);
                node->none_bitmap = new_bitmap(bitmap_size);
                node->child_bitmap = new_bitmap(bitmap_size);
                memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
                memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);

                for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size; ) {
                    int next = offset + 1, next_i = -1;
                    while (next < size) {
                        next_i = PREDICT_POS(node, keys[next]);
                        if (next_i == item_i) {
                            next ++;
                        } else {
                            break;
                        }
                    }
                    if (next == offset + 1) {
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        node->items[item_i].comp.data.key = keys[offset];
                        node->items[item_i].comp.data.value = values[offset];
                    } else {
                        // ASSERT(next - offset <= (size+2) / 3);
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        BITMAP_SET(node->child_bitmap, item_i);
                        node->items[item_i].comp.child = new_nodes(1);
                        s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});
                    }
                    if (next >= size) {
                        break;
                    } else {
                        item_i = next_i;
                        offset = next;
                    }
                }
            }
        }

        return ret;
    }

    void destory_pending()
    {
        while (!pending_two.empty()) {
            Node* node = pending_two.top(); pending_two.pop();

            delete_items(node->items, node->num_items);
            const int bitmap_size = BITMAP_SIZE(node->num_items);
            delete_bitmap(node->none_bitmap, bitmap_size);
            delete_bitmap(node->child_bitmap, bitmap_size);
            delete_nodes(node, 1);
        }
    }

    void destroy_tree(Node* root)
    {
        std::stack<Node*> s;
        s.push(root);
        while (!s.empty()) {
            Node* node = s.top(); s.pop();

            if (node->is_compact) {
                stats.compact_leaf_nodes--;
                delete_items(node->items, node->num_items);
                delete_nodes(node, 1);
                continue;
            }

            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                }
            }

            if (node->is_two) {
                LIPPHYBRID_RT_ASSERT(node->build_size == 2);
                LIPPHYBRID_RT_ASSERT(node->num_items == 8);
                node->size = 2;
                node->num_inserts = node->num_insert_to_data = 0;
                node->none_bitmap[0] = 0xff;
                node->child_bitmap[0] = 0;
                pending_two.push(node);
            } else {
                delete_items(node->items, node->num_items);
                const int bitmap_size = BITMAP_SIZE(node->num_items);
                delete_bitmap(node->none_bitmap, bitmap_size);
                delete_bitmap(node->child_bitmap, bitmap_size);
                delete_nodes(node, 1);
            }
        }
    }

    void scan_and_destory_tree(Node* _root, T* keys, P* values, bool destory = true)
    {
        typedef std::pair<int, Node*> Segment; // <begin, Node*>
        std::stack<Segment> s;

        s.push(Segment(0, _root));
        while (!s.empty()) {
            int begin = s.top().first;
            Node* node = s.top().second;
            const int SHOULD_END_POS = begin + node->size;
            s.pop();

            if (node->is_compact) {
                for (int i = 0; i < node->size; i++) {
                    keys[begin] = node->items[i].comp.data.key;
                    values[begin] = node->items[i].comp.data.value;
                    begin++;
                }
            } else {
                for (int i = 0; i < node->num_items; i ++) {
                    if (BITMAP_GET(node->none_bitmap, i) == 0) {
                        if (BITMAP_GET(node->child_bitmap, i) == 0) {
                            keys[begin] = node->items[i].comp.data.key;
                            values[begin] = node->items[i].comp.data.value;
                            begin ++;
                        } else {
                            s.push(Segment(begin, node->items[i].comp.child));
                            begin += node->items[i].comp.child->size;
                        }
                    }
                }
            }
            LIPPHYBRID_RT_ASSERT(SHOULD_END_POS == begin);

            if (destory) {
                if (node->is_compact) {
                    stats.compact_leaf_nodes--;
                    delete_items(node->items, node->num_items);
                    delete_nodes(node, 1);
                } else if (node->is_two) {
                    LIPPHYBRID_RT_ASSERT(node->build_size == 2);
                    LIPPHYBRID_RT_ASSERT(node->num_items == 8);
                    node->size = 2;
                    node->num_inserts = node->num_insert_to_data = 0;
                    node->none_bitmap[0] = 0xff;
                    node->child_bitmap[0] = 0;
                    pending_two.push(node);
                } else {
                    delete_items(node->items, node->num_items);
                    const int bitmap_size = BITMAP_SIZE(node->num_items);
                    delete_bitmap(node->none_bitmap, bitmap_size);
                    delete_bitmap(node->child_bitmap, bitmap_size);
                    delete_nodes(node, 1);
                }
            }
        }
    }

    Node* insert_tree(Node* _node, const T& key, const P& value, bool* ok = nullptr)
    {
        constexpr int MAX_DEPTH = 128;
        Node* path[MAX_DEPTH];
        int path_size = 0;
        int insert_to_data = 0;

        for (Node* node = _node; ; ) {
            LIPPHYBRID_RT_ASSERT(path_size < MAX_DEPTH);
            path[path_size ++] = node;

            node->size ++;
            node->num_inserts ++;
            if (node->is_compact) {
                const int old_size = node->size - 1;
                // 注意：上面已经先把 size 加一了，但新尾部槽位还没有初始化。
                // 因此插入定位只能在 old_size 个有效元素里二分，否则会偶发读到未初始化 key。
                const int pos = compact_lower_bound(node, key, old_size);
                if (pos < old_size && node->items[pos].comp.data.key == key) {
                    node->items[pos].comp.data.value = value;
                    for (int i = 0; i < path_size; i++) {
                        path[i]->size--;
                    }
                    if (ok) {
                        *ok = false;
                    }
                    return path[0];
                }

                if (node->size > node->num_items) {
                    const int new_capacity = std::max(node->num_items * 2, node->size);
                    Item* new_items_ptr = new_items(new_capacity);
                    for (int i = 0; i < pos; i++) {
                        new_items_ptr[i] = node->items[i];
                    }
                    new_items_ptr[pos].comp.data.key = key;
                    new_items_ptr[pos].comp.data.value = value;
                    for (int i = pos; i < old_size; i++) {
                        new_items_ptr[i + 1] = node->items[i];
                    }
                    delete_items(node->items, node->num_items);
                    node->items = new_items_ptr;
                    node->num_items = new_capacity;
                } else {
                    for (int i = old_size; i > pos; i--) {
                        node->items[i] = node->items[i - 1];
                    }
                    node->items[pos].comp.data.key = key;
                    node->items[pos].comp.data.value = value;
                }
                break;
            }
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                BITMAP_CLEAR(node->none_bitmap, pos);
                node->items[pos].comp.data.key = key;
                node->items[pos].comp.data.value = value;
                break;
            } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                BITMAP_SET(node->child_bitmap, pos);
                node->items[pos].comp.child = build_tree_two(key, value, node->items[pos].comp.data.key, node->items[pos].comp.data.value);
                insert_to_data = 1;
                break;
            } else {
                node = node->items[pos].comp.child;
            }
        }
        for (int i = 0; i < path_size; i ++) {
            path[i]->num_insert_to_data += insert_to_data;
        }

        for (int i = 0; i < path_size; i ++) {
            Node* node = path[i];
            const int num_inserts = node->num_inserts;
            const int num_insert_to_data = node->num_insert_to_data;
            const bool need_rebuild = node->fixed == 0 && node->size >= node->build_size * 2 && node->size >= 64 && num_insert_to_data * 10 >= num_inserts;

            if (need_rebuild) {
                const int ESIZE = node->size;
                T* keys = new T[ESIZE];
                P* values = new P[ESIZE];

                #if COLLECT_TIME
                auto start_time_scan = std::chrono::high_resolution_clock::now();
                #endif
                scan_and_destory_tree(node, keys, values);
                #if COLLECT_TIME
                auto end_time_scan = std::chrono::high_resolution_clock::now();
                auto duration_scan = end_time_scan - start_time_scan;
                stats.time_scan_and_destory_tree += std::chrono::duration_cast<std::chrono::nanoseconds>(duration_scan).count() * 1e-9;
                #endif

                #if COLLECT_TIME
                auto start_time_build = std::chrono::high_resolution_clock::now();
                #endif
                Node* new_node = build_tree_bulk(keys, values, ESIZE);
                #if COLLECT_TIME
                auto end_time_build = std::chrono::high_resolution_clock::now();
                auto duration_build = end_time_build - start_time_build;
                stats.time_build_tree_bulk += std::chrono::duration_cast<std::chrono::nanoseconds>(duration_build).count() * 1e-9;
                #endif

                delete[] keys;
                delete[] values;

                path[i] = new_node;
                if (i > 0) {
                    int pos = PREDICT_POS(path[i-1], key);
                    path[i-1]->items[pos].comp.child = new_node;
                }

                break;
            }
        }
        if(ok) {
            *ok = true;
        }
        return path[0];
    }

    // SATISFY_LOWER = true means all the keys in the subtree of `node` are no less than to `lower`.
    template<bool SATISFY_LOWER>
    int range_core_len(std::pair <T, P> *results, int pos, Node *node, const T &lower, int len) {
        if (node->is_compact) {
            int i = 0;
            if constexpr(!SATISFY_LOWER) {
                i = compact_lower_bound(node, lower);
            }
            while (i < node->size && pos < len) {
                results[pos] = {node->items[i].comp.data.key, node->items[i].comp.data.value};
                pos++;
                i++;
            }
            return pos;
        }
        if constexpr(SATISFY_LOWER)
        {
            int bit_pos = 0;
            const bitmap_t *none_bitmap = node->none_bitmap;
            while (bit_pos < node->num_items) {
                bitmap_t not_none = ~(*none_bitmap);
                while (not_none) {
                    int latest_pos = BITMAP_NEXT_1(not_none);
                    not_none ^= 1 << latest_pos;

                    int i = bit_pos + latest_pos;
                    if (BITMAP_GET(node->child_bitmap, i) == 0) {
                        results[pos] = {node->items[i].comp.data.key, node->items[i].comp.data.value};
                        // __builtin_prefetch((void*)&(node->items[i].comp.data.key) + 64);
                        pos++;
                    } else {
                        pos = range_core_len<true>(results, pos, node->items[i].comp.child, lower, len);
                    }
                    if (pos >= len) {
                        return pos;
                    }
                }

                bit_pos += BITMAP_WIDTH;
                none_bitmap++;
            }
            return pos;
        } else {
            int lower_pos = PREDICT_POS(node, lower);
            if (BITMAP_GET(node->none_bitmap, lower_pos) == 0) {
                if (BITMAP_GET(node->child_bitmap, lower_pos) == 0) {
                    if (node->items[lower_pos].comp.data.key >= lower) {
                        results[pos] = {node->items[lower_pos].comp.data.key, node->items[lower_pos].comp.data.value};
                        pos++;
                    }
                } else {
                    pos = range_core_len<false>(results, pos, node->items[lower_pos].comp.child, lower, len);
                }
                if (pos >= len) {
                    return pos;
                }
            }
            if (lower_pos + 1 >= node->num_items) {
                return pos;
            }
            int bit_pos = (lower_pos + 1) / BITMAP_WIDTH * BITMAP_WIDTH;
            const bitmap_t *none_bitmap = node->none_bitmap + bit_pos / BITMAP_WIDTH;
            while (bit_pos < node->num_items) {
                bitmap_t not_none = ~(*none_bitmap);
                while (not_none) {
                    int latest_pos = BITMAP_NEXT_1(not_none);
                    not_none ^= 1 << latest_pos;

                    int i = bit_pos + latest_pos;
                    if (i <= lower_pos) continue;
                    if (BITMAP_GET(node->child_bitmap, i) == 0) {
                        results[pos] = {node->items[i].comp.data.key, node->items[i].comp.data.value};
                        // __builtin_prefetch((void*)&(node->items[i].comp.data.key) + 64);
                        pos++;
                    } else {
                        pos = range_core_len<true>(results, pos, node->items[i].comp.child, lower, len);
                    }
                    if (pos >= len) {
                        return pos;
                    }
                }
                bit_pos += BITMAP_WIDTH;
                none_bitmap++;
            }
            return pos;
        }
    }
};

#endif // __LIPPHYBRID_H__
