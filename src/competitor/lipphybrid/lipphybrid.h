#include "./src/src/core/lipphybrid.h"
#include "../indexInterface.h"

// LIPP-Hybrid 的第一阶段实验包装层。
// 当前版本先保留 LIPP 的节点结构和 API，只在建树/重建时加入“自适应单调核函数”。
// 后续再逐步把叶节点改为紧凑布局，并把粗粒度重建统计接入这里。
template<class KEY_TYPE, class PAYLOAD_TYPE>
class LIPPHybridInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
    void init(Param *param = nullptr) {}

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

private:
    LIPPHybrid <KEY_TYPE, PAYLOAD_TYPE> lipphybrid;
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(
    std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param) {
    lipphybrid.bulk_load(key_value, static_cast<int>(num));
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
    bool exist;
    val = lipphybrid.at(key, false, exist);
    return exist;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    return lipphybrid.insert(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    return lipphybrid.update(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
    return lipphybrid.remove(key);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t LIPPHybridInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(
    KEY_TYPE key_low_bound, size_t key_num, std::pair <KEY_TYPE, PAYLOAD_TYPE> *result, Param *param) {
    if(!result) {
        result = new std::pair <KEY_TYPE, PAYLOAD_TYPE>[key_num];
    }
    return lipphybrid.range_query_len(result, key_low_bound, key_num);
}
