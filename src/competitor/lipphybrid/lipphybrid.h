#include "./src/src/core/lipphybrid.h"
#include "../indexInterface.h"

// LIPP-Hybrid 的实验包装层。
// 目前支持两条消融线：自适应单调核函数，以及紧凑叶节点。
// 原始 lipp 类型不经过这里，方便和实验版本做公平对照。
template<class KEY_TYPE, class PAYLOAD_TYPE,
         LIPPHybridKernelPolicy POLICY = LIPPHybridKernelPolicy::FALLBACK_ONLY,
         LIPPHybridCompactPolicy COMPACT_POLICY = LIPPHybridCompactPolicy::DISABLED>
class LIPPHybridInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
    LIPPHybridInterface() : lipphybrid(0, true, POLICY, COMPACT_POLICY) {}

    void init(Param *param = nullptr) {}

    void set_workload(double read_ratio, double insert_ratio, double update_ratio,
                      double scan_ratio, double delete_ratio, long long operations_num,
                      long long scan_num) {
        lipphybrid.set_workload(read_ratio, insert_ratio, update_ratio,
                                scan_ratio, delete_ratio, operations_num, scan_num);
    }

    void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr);

    bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

    bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

    bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

    bool remove(KEY_TYPE key, Param *param = nullptr);

    size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair <KEY_TYPE, PAYLOAD_TYPE> *result,
                Param *param = nullptr);

    long long memory_consumption() { return lipphybrid.total_size(); }

    // 复用 GRE 现有 CSV 列做第一阶段消融：
    // num_expand_and_scales      -> 选择非线性核函数的节点总数
    // num_expand_and_retrains    -> 选择 log1p 的节点数
    // num_downward_splits        -> 选择 sqrt 的节点数
    // num_sideways_splits        -> 选择 cbrt 的节点数
    long long num_expand_and_scales() { return lipphybrid.adaptive_kernel_nodes(); }
    long long num_expand_and_retrains() { return lipphybrid.adaptive_kernel_log1p_nodes(); }
    long long num_downward_splits() { return lipphybrid.adaptive_kernel_sqrt_nodes(); }
    long long num_sideways_splits() { return lipphybrid.adaptive_kernel_cbrt_nodes(); }
    long long kernel_nodes() { return lipphybrid.adaptive_kernel_nodes(); }
    long long log1p_kernel_nodes() { return lipphybrid.adaptive_kernel_log1p_nodes(); }
    long long sqrt_kernel_nodes() { return lipphybrid.adaptive_kernel_sqrt_nodes(); }
    long long cbrt_kernel_nodes() { return lipphybrid.adaptive_kernel_cbrt_nodes(); }
    long long hardness_sqrt_enabled() { return lipphybrid.hardness_sqrt_enabled(); }
    long long adaptive_compact_enabled() { return lipphybrid.adaptive_compact_enabled_stat(); }
    long long adaptive_compact_candidate_nodes() { return lipphybrid.adaptive_compact_candidate_nodes(); }
    long long adaptive_compact_estimated_saved_bytes() { return lipphybrid.adaptive_compact_estimated_saved_bytes(); }
    long long adaptive_compact_estimated_extra_access_bytes() { return lipphybrid.adaptive_compact_estimated_extra_access_bytes(); }
    long long compact_leaf_nodes() { return lipphybrid.compact_leaf_nodes(); }

private:
    LIPPHybrid <KEY_TYPE, PAYLOAD_TYPE> lipphybrid;
};

template<class KEY_TYPE, class PAYLOAD_TYPE, LIPPHybridKernelPolicy POLICY, LIPPHybridCompactPolicy COMPACT_POLICY>
void LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE, POLICY, COMPACT_POLICY>::bulk_load(
    std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param) {
    lipphybrid.bulk_load(key_value, static_cast<int>(num));
}

template<class KEY_TYPE, class PAYLOAD_TYPE, LIPPHybridKernelPolicy POLICY, LIPPHybridCompactPolicy COMPACT_POLICY>
bool LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE, POLICY, COMPACT_POLICY>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
    bool exist;
    val = lipphybrid.at(key, false, exist);
    return exist;
}

template<class KEY_TYPE, class PAYLOAD_TYPE, LIPPHybridKernelPolicy POLICY, LIPPHybridCompactPolicy COMPACT_POLICY>
bool LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE, POLICY, COMPACT_POLICY>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    return lipphybrid.insert(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE, LIPPHybridKernelPolicy POLICY, LIPPHybridCompactPolicy COMPACT_POLICY>
bool LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE, POLICY, COMPACT_POLICY>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    return lipphybrid.update(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE, LIPPHybridKernelPolicy POLICY, LIPPHybridCompactPolicy COMPACT_POLICY>
bool LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE, POLICY, COMPACT_POLICY>::remove(KEY_TYPE key, Param *param) {
    return lipphybrid.remove(key);
}

template<class KEY_TYPE, class PAYLOAD_TYPE, LIPPHybridKernelPolicy POLICY, LIPPHybridCompactPolicy COMPACT_POLICY>
size_t LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE, POLICY, COMPACT_POLICY>::scan(
    KEY_TYPE key_low_bound, size_t key_num, std::pair <KEY_TYPE, PAYLOAD_TYPE> *result, Param *param) {
    if(!result) {
        result = new std::pair <KEY_TYPE, PAYLOAD_TYPE>[key_num];
    }
    return lipphybrid.range_query_len(result, key_low_bound, key_num);
}
