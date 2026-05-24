#include <iomanip>

#pragma once

struct Param { // for xindex
  size_t worker_num;
  uint32_t thread_id;

  Param(size_t worker_num, uint32_t thread_id) : worker_num(worker_num), thread_id(thread_id) {}
};

struct BaseCompare {
  template<class T1, class T2>
  bool operator()(const T1 &x, const T2 &y) const {
    static_assert(
      std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
      "Comparison types must be numeric.");
    return x < y;
  }
};

template<class KEY_TYPE, class PAYLOAD_TYPE, class KeyComparator=BaseCompare>
class indexInterface {
public:
  virtual void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr) = 0;

  virtual bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) = 0;

  virtual bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) = 0;

  virtual bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) = 0;

  virtual bool remove(KEY_TYPE key, Param *param = nullptr) = 0;

  virtual size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                      Param *param = nullptr) = 0;

  virtual void init(Param *param = nullptr) = 0;

  // benchmark 在 bulk_load 前传入 workload。默认实现为空，只有需要自适应建树的索引用它。
  virtual void set_workload(double read_ratio, double insert_ratio, double update_ratio,
                            double scan_ratio, double delete_ratio, long long operations_num,
                            long long scan_num) {}

  virtual long long memory_consumption() = 0; // bytes

  // 下面这些统计项主要用于 ALEX。其他索引默认返回 0。
  // 用虚函数放在统一接口里，是为了 benchmark 可以不关心具体索引类型。
  virtual long long num_expand_and_scales() { return 0; }

  virtual long long num_expand_and_retrains() { return 0; }

  virtual long long num_downward_splits() { return 0; }

  virtual long long num_sideways_splits() { return 0; }

  virtual long long num_model_node_splits() { return 0; }

  // LIPP-Hybrid 专用统计项。其他索引默认返回 0，方便 benchmark 统一输出。
  virtual long long kernel_nodes() { return 0; }

  virtual long long log1p_kernel_nodes() { return 0; }

  virtual long long sqrt_kernel_nodes() { return 0; }

  virtual long long cbrt_kernel_nodes() { return 0; }

  virtual long long hardness_sqrt_enabled() { return 0; }

  virtual long long adaptive_compact_enabled() { return 0; }

  virtual long long adaptive_compact_candidate_nodes() { return 0; }

  virtual long long adaptive_compact_estimated_saved_bytes() { return 0; }

  virtual long long adaptive_compact_estimated_extra_access_bytes() { return 0; }

  virtual long long compact_leaf_nodes() { return 0; }
};
