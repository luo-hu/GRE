#include"./src/src/core/alex.h"
#include"../indexInterface.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class alexInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  void init(Param *param = nullptr) {}

  void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr);

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool remove(KEY_TYPE key, Param *param = nullptr);

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
              Param *param = nullptr);

  long long memory_consumption() { return index.model_size() + index.data_size(); }

  // ALEX 内部统计：数据节点扩容并只做线性缩放的次数。
  long long num_expand_and_scales() { return index.get_stats().num_expand_and_scales; }

  // ALEX 内部统计：数据节点扩容并重新训练模型的次数。
  long long num_expand_and_retrains() { return index.get_stats().num_expand_and_retrains; }

  // ALEX 内部统计：向下分裂次数，表示局部模型需要更深层结构。
  long long num_downward_splits() { return index.get_stats().num_downward_splits; }

  // ALEX 内部统计：横向分裂次数，表示某个数据节点需要拆成并列节点。
  long long num_sideways_splits() { return index.get_stats().num_sideways_splits; }

  // ALEX 内部统计：模型节点分裂次数。
  long long num_model_node_splits() { return index.get_stats().num_model_node_splits; }

private:
  alex::Alex<KEY_TYPE, PAYLOAD_TYPE, alex::AlexCompare, std::allocator < std::pair < KEY_TYPE, PAYLOAD_TYPE>>, false>
  index;
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void alexInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num,
                                                      Param *param) {
  index.bulk_load(key_value, (int) num);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool alexInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
  PAYLOAD_TYPE *res = index.get_payload(key);
  if (res != nullptr) {
    val = *res;
    return true;
  }
  return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool alexInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  return index.insert(key, value).second;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool alexInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    // return index.update(key, value);
    return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool alexInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
  auto num_erase = index.erase(key);
  return num_erase > 0;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t alexInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                   std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                                                   Param *param) {
  auto iter = index.lower_bound(key_low_bound);
  int scan_size = 0;
  for (scan_size = 0; scan_size < key_num && !iter.is_end(); scan_size++) {
    result[scan_size] = {(*iter).first, (*iter).second};
    iter++;
  }
  return scan_size;
}
