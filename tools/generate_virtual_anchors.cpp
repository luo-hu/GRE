#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct Config {
  std::string input_path;
  std::string output_path;
  std::string policy = "hard";
  std::string source = "full";
  double ratio = 0.01;
  double init_table_ratio = 0.5;
  size_t seed = 1866;
  size_t window_size = 4096;
};

struct WindowScore {
  size_t begin = 0;
  size_t end = 0;
  double score = 0.0;
};

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " --input_keys=PATH --output_keys=PATH "
      << "[--ratio=0.01] [--policy=hard|guarded|guarded_v2|gap_only|cliff_only|gap|uniform] "
      << "[--source=full|init] [--init_table_ratio=0.5] [--seed=1866] "
      << "[--window_size=4096]\n";
}

std::string value_of(const std::string& arg) {
  auto pos = arg.find('=');
  if (pos == std::string::npos) {
    return "";
  }
  return arg.substr(pos + 1);
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--input_keys=", 0) == 0) {
      cfg.input_path = value_of(arg);
    } else if (arg.rfind("--output_keys=", 0) == 0) {
      cfg.output_path = value_of(arg);
    } else if (arg.rfind("--ratio=", 0) == 0) {
      cfg.ratio = std::stod(value_of(arg));
    } else if (arg.rfind("--policy=", 0) == 0) {
      cfg.policy = value_of(arg);
    } else if (arg.rfind("--source=", 0) == 0) {
      cfg.source = value_of(arg);
    } else if (arg.rfind("--window_size=", 0) == 0) {
      cfg.window_size = std::stoull(value_of(arg));
    } else if (arg.rfind("--init_table_ratio=", 0) == 0) {
      cfg.init_table_ratio = std::stod(value_of(arg));
    } else if (arg.rfind("--seed=", 0) == 0) {
      cfg.seed = std::stoull(value_of(arg));
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      usage(argv[0]);
      std::exit(1);
    }
  }

  if (cfg.input_path.empty() || cfg.output_path.empty()) {
    usage(argv[0]);
    std::exit(1);
  }
  if (!(cfg.ratio >= 0.0 && cfg.ratio <= 1.0)) {
    std::cerr << "--ratio must be in [0, 1]\n";
    std::exit(1);
  }
  if (cfg.source != "full" && cfg.source != "init") {
    std::cerr << "--source must be full or init\n";
    std::exit(1);
  }
  if (!(cfg.init_table_ratio > 0.0 && cfg.init_table_ratio <= 1.0)) {
    std::cerr << "--init_table_ratio must be in (0, 1]\n";
    std::exit(1);
  }
  if (cfg.policy != "hard" && cfg.policy != "guarded" && cfg.policy != "guarded_v2" &&
      cfg.policy != "gap_only" && cfg.policy != "cliff_only" &&
      cfg.policy != "gap" && cfg.policy != "uniform") {
    std::cerr << "--policy must be hard, guarded, guarded_v2, gap_only, cliff_only, gap, or uniform\n";
    std::exit(1);
  }
  if (cfg.window_size < 16) {
    std::cerr << "--window_size must be at least 16\n";
    std::exit(1);
  }
  return cfg;
}

std::vector<uint64_t> read_binary_keys(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open input keys: " + path);
  }
  uint64_t n = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof(uint64_t));
  std::vector<uint64_t> keys(n);
  in.read(reinterpret_cast<char*>(keys.data()),
          static_cast<std::streamsize>(n * sizeof(uint64_t)));
  if (!in && n > 0) {
    throw std::runtime_error("failed to read all keys from: " + path);
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

std::vector<uint64_t> select_anchor_source_keys(std::vector<uint64_t> keys,
                                                const Config& cfg) {
  if (cfg.source == "full") {
    return keys;
  }

  // fair/init 模式必须复现 microbench 的初始化方式：
  // 先对去重后的全量 key 用同一个 seed shuffle，再取前 init_table_ratio 作为 bulk-load key。
  std::mt19937 gen(cfg.seed);
  std::shuffle(keys.begin(), keys.end(), gen);
  size_t init_size = static_cast<size_t>(cfg.init_table_ratio * keys.size());
  init_size = std::max<size_t>(1, std::min(init_size, keys.size()));
  keys.resize(init_size);
  std::sort(keys.begin(), keys.end());
  return keys;
}

void write_binary_keys(const std::string& path, const std::vector<uint64_t>& keys) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("cannot open output keys: " + path);
  }
  uint64_t n = keys.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(uint64_t));
  out.write(reinterpret_cast<const char*>(keys.data()),
            static_cast<std::streamsize>(n * sizeof(uint64_t)));
}

bool add_anchor(uint64_t key, const std::unordered_set<uint64_t>& real_keys,
                std::unordered_set<uint64_t>& used,
                std::vector<uint64_t>& anchors, size_t budget) {
  if (anchors.size() >= budget || real_keys.count(key) || used.count(key)) {
    return false;
  }
  used.insert(key);
  anchors.push_back(key);
  return true;
}

uint64_t midpoint_anchor(uint64_t left, uint64_t right,
                         const std::unordered_set<uint64_t>& real_keys,
                         const std::unordered_set<uint64_t>& used) {
  if (right <= left + 1) {
    return 0;
  }

  uint64_t gap = right - left;
  uint64_t mid = left + gap / 2;
  if (!real_keys.count(mid) && !used.count(mid)) {
    return mid;
  }

  // 如果中点恰好被占用，就从中点向两侧试探几个位置，避免和真实 key 重复。
  for (uint64_t step = 1; step < 64 && step < gap; ++step) {
    if (mid >= left + 1 + step) {
      uint64_t candidate = mid - step;
      if (!real_keys.count(candidate) && !used.count(candidate)) {
        return candidate;
      }
    }
    if (mid + step < right) {
      uint64_t candidate = mid + step;
      if (!real_keys.count(candidate) && !used.count(candidate)) {
        return candidate;
      }
    }
  }
  return 0;
}

void add_from_gap(uint64_t left, uint64_t right, const std::unordered_set<uint64_t>& real_keys,
                  std::unordered_set<uint64_t>& used, std::vector<uint64_t>& anchors,
                  size_t budget) {
  if (right <= left + 1 || anchors.size() >= budget) {
    return;
  }

  uint64_t candidate = midpoint_anchor(left, right, real_keys, used);
  if (candidate != 0) {
    add_anchor(candidate, real_keys, used, anchors, budget);
  }
}

std::vector<uint64_t> largest_gap_anchors(const std::vector<uint64_t>& keys, size_t budget,
                                          size_t begin, size_t end,
                                          const std::unordered_set<uint64_t>& real_keys,
                                          std::unordered_set<uint64_t>& used) {
  std::vector<std::pair<uint64_t, size_t>> gaps;
  for (size_t i = begin; i + 1 < end; ++i) {
    uint64_t gap = keys[i + 1] - keys[i];
    if (gap > 1) {
      gaps.push_back({gap, i});
    }
  }
  std::sort(gaps.begin(), gaps.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<uint64_t> anchors;
  anchors.reserve(budget);
  for (const auto& item : gaps) {
    add_from_gap(keys[item.second], keys[item.second + 1], real_keys, used, anchors, budget);
    if (anchors.size() >= budget) {
      break;
    }
  }
  return anchors;
}

double local_sse_score(const std::vector<uint64_t>& keys, size_t begin, size_t end) {
  const size_t n = end - begin;
  if (n < 2 || keys[end - 1] == keys[begin]) {
    return 0.0;
  }

  long double sum_x = 0.0;
  long double sum_y = 0.0;
  long double sum_xx = 0.0;
  long double sum_xy = 0.0;
  uint64_t min_gap = std::numeric_limits<uint64_t>::max();
  uint64_t max_gap = 1;

  const long double span = static_cast<long double>(keys[end - 1] - keys[begin]);
  for (size_t i = begin; i < end; ++i) {
    long double x = static_cast<long double>(keys[i] - keys[begin]) / span;
    long double y = static_cast<long double>(i - begin) / static_cast<long double>(n - 1);
    sum_x += x;
    sum_y += y;
    sum_xx += x * x;
    sum_xy += x * y;
    if (i + 1 < end) {
      uint64_t gap = keys[i + 1] - keys[i];
      if (gap > 0) {
        min_gap = std::min(min_gap, gap);
        max_gap = std::max(max_gap, gap);
      }
    }
  }

  long double denom = n * sum_xx - sum_x * sum_x;
  if (std::fabs(denom) < 1e-18L) {
    return 0.0;
  }
  long double a = (n * sum_xy - sum_x * sum_y) / denom;
  long double b = (sum_y - a * sum_x) / n;
  long double sse = 0.0;
  for (size_t i = begin; i < end; ++i) {
    long double x = static_cast<long double>(keys[i] - keys[begin]) / span;
    long double y = static_cast<long double>(i - begin) / static_cast<long double>(n - 1);
    long double err = a * x + b - y;
    sse += err * err;
  }

  // 这里的 hard score 是离线启发式：局部线性拟合误差越大、gap 变化越剧烈，
  // 说明 Z-order CDF 越难被一条局部线性模型拟合。
  double mse = static_cast<double>(sse / n);
  double gap_factor = 1.0;
  if (min_gap != std::numeric_limits<uint64_t>::max() && min_gap > 0) {
    gap_factor += std::log(static_cast<double>(max_gap) / static_cast<double>(min_gap));
  }
  return mse * gap_factor;
}

double local_sse_score_with_anchor(const std::vector<uint64_t>& keys, size_t begin, size_t end,
                                   uint64_t anchor) {
  std::vector<uint64_t> local;
  local.reserve(end - begin + 1);
  for (size_t i = begin; i < end; ++i) {
    local.push_back(keys[i]);
  }
  local.push_back(anchor);
  std::sort(local.begin(), local.end());
  local.erase(std::unique(local.begin(), local.end()), local.end());
  return local_sse_score(local, 0, local.size());
}

std::vector<WindowScore> score_windows(const std::vector<uint64_t>& keys, size_t window_size) {
  std::vector<WindowScore> windows;
  for (size_t begin = 0; begin < keys.size(); begin += window_size) {
    size_t end = std::min(keys.size(), begin + window_size);
    if (end - begin >= 16) {
      windows.push_back({begin, end, local_sse_score(keys, begin, end)});
    }
  }
  return windows;
}

uint64_t percentile_uint64(std::vector<uint64_t> values, double q) {
  if (values.empty()) {
    return 0;
  }
  std::sort(values.begin(), values.end());
  q = std::max(0.0, std::min(1.0, q));
  size_t pos = static_cast<size_t>(q * (values.size() - 1));
  return values[pos];
}

std::vector<uint64_t> uniform_anchors(const std::vector<uint64_t>& keys, size_t budget,
                                      const std::unordered_set<uint64_t>& real_keys) {
  std::vector<uint64_t> anchors;
  anchors.reserve(budget);
  std::unordered_set<uint64_t> used;
  if (keys.empty() || budget == 0 || keys.back() <= keys.front() + 1) {
    return anchors;
  }
  long double min_key = keys.front();
  long double span = static_cast<long double>(keys.back() - keys.front());
  for (size_t i = 1; i <= budget && anchors.size() < budget; ++i) {
    auto candidate = static_cast<uint64_t>(min_key + span * i / (budget + 1));
    add_anchor(candidate, real_keys, used, anchors, budget);
  }
  std::sort(anchors.begin(), anchors.end());
  return anchors;
}

std::vector<uint64_t> gap_anchors(const std::vector<uint64_t>& keys, size_t budget,
                                  const std::unordered_set<uint64_t>& real_keys) {
  std::unordered_set<uint64_t> used;
  auto anchors = largest_gap_anchors(keys, budget, 0, keys.size(), real_keys, used);
  std::sort(anchors.begin(), anchors.end());
  return anchors;
}

std::vector<uint64_t> guarded_anchors(const std::vector<uint64_t>& keys, size_t budget,
                                      size_t window_size,
                                      const std::unordered_set<uint64_t>& real_keys) {
  struct Candidate {
    uint64_t gap = 0;
    size_t left_pos = 0;
  };

  std::vector<uint64_t> anchors;
  anchors.reserve(budget);
  if (keys.size() < 2 || budget == 0) {
    return anchors;
  }

  std::vector<Candidate> candidates;
  candidates.reserve(keys.size());
  for (size_t i = 0; i + 1 < keys.size(); ++i) {
    uint64_t gap = keys[i + 1] - keys[i];
    if (gap > 1) {
      candidates.push_back({gap, i});
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.gap > b.gap; });

  std::unordered_set<uint64_t> used;
  std::vector<size_t> window_anchor_count((keys.size() + window_size - 1) / window_size, 0);
  const size_t per_window_cap = std::max<size_t>(1, window_size / 4096);
  const double min_relative_improvement = 0.005;

  for (const auto& candidate : candidates) {
    if (anchors.size() >= budget) {
      break;
    }

    size_t window_id = candidate.left_pos / window_size;
    if (window_id >= window_anchor_count.size() ||
        window_anchor_count[window_id] >= per_window_cap) {
      continue;
    }

    size_t half_window = window_size / 2;
    size_t begin = candidate.left_pos > half_window ? candidate.left_pos - half_window : 0;
    size_t end = std::min(keys.size(), candidate.left_pos + 1 + half_window);
    if (end - begin < 16) {
      continue;
    }

    uint64_t anchor = midpoint_anchor(keys[candidate.left_pos], keys[candidate.left_pos + 1],
                                      real_keys, used);
    if (anchor == 0) {
      continue;
    }

    double old_score = local_sse_score(keys, begin, end);
    double new_score = local_sse_score_with_anchor(keys, begin, end, anchor);
    double improvement = old_score - new_score;
    double threshold = std::max(1e-12, old_score * min_relative_improvement);

    // guarded 的关键：只接受能降低局部拟合难度的锚点。
    // 这样可以避免 hard 策略那种“难就多塞”的过度干预。
    if (improvement > threshold &&
        add_anchor(anchor, real_keys, used, anchors, budget)) {
      window_anchor_count[window_id]++;
    }
  }

  std::sort(anchors.begin(), anchors.end());
  return anchors;
}

std::vector<uint64_t> guarded_v2_anchors(const std::vector<uint64_t>& keys, size_t budget,
                                         size_t window_size,
                                         const std::unordered_set<uint64_t>& real_keys,
                                         bool use_gap_candidates,
                                         bool use_cliff_candidates) {
  struct Candidate {
    double priority = 0.0;
    size_t left_pos = 0;
  };

  std::vector<uint64_t> anchors;
  anchors.reserve(budget);
  if (keys.size() < 2 || budget == 0) {
    return anchors;
  }

  const size_t num_windows = (keys.size() + window_size - 1) / window_size;
  const size_t dense_run = std::max<size_t>(32, std::min<size_t>(128, window_size / 64));

  std::vector<uint64_t> gaps;
  gaps.reserve(keys.size());
  for (size_t i = 0; i + 1 < keys.size(); ++i) {
    uint64_t gap = keys[i + 1] - keys[i];
    if (gap > 0) {
      gaps.push_back(gap);
    }
  }
  uint64_t large_gap_threshold = std::max<uint64_t>(2, percentile_uint64(gaps, 0.90));

  std::vector<uint64_t> dense_spans;
  if (keys.size() > dense_run) {
    dense_spans.reserve(keys.size() - dense_run);
    for (size_t i = 0; i + dense_run < keys.size(); ++i) {
      dense_spans.push_back(keys[i + dense_run] - keys[i]);
    }
  }
  uint64_t dense_span_threshold = percentile_uint64(dense_spans, 0.05);

  std::vector<Candidate> candidates;
  candidates.reserve(keys.size());
  for (size_t i = 0; i + 1 < keys.size(); ++i) {
    uint64_t gap = keys[i + 1] - keys[i];
    if (use_gap_candidates && gap >= large_gap_threshold && gap > 1) {
      candidates.push_back({static_cast<double>(gap), i});
    }

    // density-aware 部分：用自适应分位数找“很多 rank 挤在很小 key span”的区域。
    // 不是在密集区内部硬塞点，而是在密集区边界的相邻 gap 上放候选锚点。
    if (use_cliff_candidates && dense_span_threshold > 0 && i + dense_run < keys.size()) {
      uint64_t span = keys[i + dense_run] - keys[i];
      if (span <= dense_span_threshold) {
        double priority = static_cast<double>(large_gap_threshold) * 2.0;
        if (i > 0 && keys[i] > keys[i - 1] + 1) {
          candidates.push_back({priority, i - 1});
        }
        if (i + dense_run + 1 < keys.size() && keys[i + dense_run + 1] > keys[i + dense_run] + 1) {
          candidates.push_back({priority, i + dense_run});
        }
        i += dense_run;
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.priority > b.priority; });

  std::unordered_set<uint64_t> used;
  std::vector<size_t> window_anchor_count(num_windows, 0);
  const double avg_budget_per_window =
      num_windows == 0 ? 0.0 : static_cast<double>(budget) / static_cast<double>(num_windows);
  const size_t per_window_cap =
      std::min<size_t>(32, std::max<size_t>(4, static_cast<size_t>(std::ceil(avg_budget_per_window * 0.25))));
  const double min_relative_improvement = 0.001;

  for (const auto& candidate : candidates) {
    if (anchors.size() >= budget) {
      break;
    }
    if (candidate.left_pos + 1 >= keys.size()) {
      continue;
    }

    size_t window_id = candidate.left_pos / window_size;
    if (window_id >= window_anchor_count.size() ||
        window_anchor_count[window_id] >= per_window_cap) {
      continue;
    }

    size_t half_window = window_size / 2;
    size_t begin = candidate.left_pos > half_window ? candidate.left_pos - half_window : 0;
    size_t end = std::min(keys.size(), candidate.left_pos + 1 + half_window);
    if (end - begin < 16) {
      continue;
    }

    uint64_t anchor = midpoint_anchor(keys[candidate.left_pos], keys[candidate.left_pos + 1],
                                      real_keys, used);
    if (anchor == 0) {
      continue;
    }

    double old_score = local_sse_score(keys, begin, end);
    double new_score = local_sse_score_with_anchor(keys, begin, end, anchor);
    double improvement = old_score - new_score;
    double threshold = std::max(1e-12, old_score * min_relative_improvement);
    if (improvement > threshold &&
        add_anchor(anchor, real_keys, used, anchors, budget)) {
      window_anchor_count[window_id]++;
    }
  }

  std::sort(anchors.begin(), anchors.end());
  return anchors;
}

std::vector<uint64_t> hard_anchors(const std::vector<uint64_t>& keys, size_t budget,
                                   size_t window_size,
                                   const std::unordered_set<uint64_t>& real_keys) {
  std::vector<uint64_t> anchors;
  anchors.reserve(budget);
  std::unordered_set<uint64_t> used;
  auto windows = score_windows(keys, window_size);
  double total_score = 0.0;
  for (const auto& w : windows) {
    total_score += w.score;
  }
  if (total_score <= 0.0) {
    return gap_anchors(keys, budget, real_keys);
  }

  std::sort(windows.begin(), windows.end(),
            [](const auto& a, const auto& b) { return a.score > b.score; });

  for (const auto& w : windows) {
    if (anchors.size() >= budget) {
      break;
    }
    size_t share = static_cast<size_t>(
        std::ceil(static_cast<double>(budget) * w.score / total_score));
    share = std::max<size_t>(1, share);
    share = std::min(share, budget - anchors.size());
    auto local = largest_gap_anchors(keys, share, w.begin, w.end, real_keys, used);
    anchors.insert(anchors.end(), local.begin(), local.end());
  }

  if (anchors.size() < budget) {
    auto more = largest_gap_anchors(keys, budget - anchors.size(), 0, keys.size(),
                                    real_keys, used);
    anchors.insert(anchors.end(), more.begin(), more.end());
  }

  std::sort(anchors.begin(), anchors.end());
  return anchors;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Config cfg = parse_args(argc, argv);
    auto all_keys = read_binary_keys(cfg.input_path);
    auto keys = select_anchor_source_keys(all_keys, cfg);
    std::unordered_set<uint64_t> real_keys;
    real_keys.reserve(keys.size() * 2);
    for (auto key : keys) {
      real_keys.insert(key);
    }

    size_t budget = static_cast<size_t>(std::llround(keys.size() * cfg.ratio));
    std::vector<uint64_t> anchors;
    if (cfg.policy == "uniform") {
      anchors = uniform_anchors(keys, budget, real_keys);
    } else if (cfg.policy == "gap") {
      anchors = gap_anchors(keys, budget, real_keys);
    } else if (cfg.policy == "guarded") {
      anchors = guarded_anchors(keys, budget, cfg.window_size, real_keys);
    } else if (cfg.policy == "guarded_v2") {
      anchors = guarded_v2_anchors(keys, budget, cfg.window_size, real_keys, true, true);
    } else if (cfg.policy == "gap_only") {
      anchors = guarded_v2_anchors(keys, budget, cfg.window_size, real_keys, true, false);
    } else if (cfg.policy == "cliff_only") {
      anchors = guarded_v2_anchors(keys, budget, cfg.window_size, real_keys, false, true);
    } else {
      anchors = hard_anchors(keys, budget, cfg.window_size, real_keys);
    }

    write_binary_keys(cfg.output_path, anchors);
    std::cout << "input_keys=" << all_keys.size() << "\n";
    std::cout << "anchor_source=" << cfg.source << "\n";
    std::cout << "source_keys=" << keys.size() << "\n";
    std::cout << "policy=" << cfg.policy << "\n";
    std::cout << "ratio=" << cfg.ratio << "\n";
    std::cout << "init_table_ratio=" << cfg.init_table_ratio << "\n";
    std::cout << "seed=" << cfg.seed << "\n";
    std::cout << "requested_budget=" << budget << "\n";
    std::cout << "written_anchors=" << anchors.size() << "\n";
    std::cout << "output_keys=" << cfg.output_path << "\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
