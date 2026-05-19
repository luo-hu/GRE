#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Config {
  // 数据分布类型：uniform 表示均匀二维点，clustered 表示高斯簇二维点。
  std::string dist = "uniform";
  // 如果提供 input_csv，则不再生成 synthetic 数据，而是读取已有 X,Y CSV。
  std::string input_csv;
  // 生成点数。
  std::size_t n = 100000;
  // clustered 模式下的簇中心数量。
  std::size_t clusters = 8;
  // clustered 模式下，每个簇周围高斯采样的标准差。
  double sigma = 0.04;
  // 固定随机种子，保证每次实验可复现。
  std::uint64_t seed = 1866;
  // 保存原始二维点，格式为 x,y，方便画图或人工检查。
  std::string csv_path;
  // 保存 GRE 可读取的一维 uint64 key 文件。
  std::string keys_path;
};

std::string require_value(const std::string &arg) {
  const auto pos = arg.find('=');
  if (pos == std::string::npos || pos + 1 == arg.size()) {
    throw std::invalid_argument("Expected --key=value, got " + arg);
  }
  return arg.substr(pos + 1);
}

Config parse_args(int argc, char **argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--dist=", 0) == 0) {
      cfg.dist = require_value(arg);
    } else if (arg.rfind("--n=", 0) == 0) {
      cfg.n = std::stoull(require_value(arg));
    } else if (arg.rfind("--clusters=", 0) == 0) {
      cfg.clusters = std::stoull(require_value(arg));
    } else if (arg.rfind("--sigma=", 0) == 0) {
      cfg.sigma = std::stod(require_value(arg));
    } else if (arg.rfind("--seed=", 0) == 0) {
      cfg.seed = std::stoull(require_value(arg));
    } else if (arg.rfind("--csv=", 0) == 0) {
      cfg.csv_path = require_value(arg);
    } else if (arg.rfind("--keys=", 0) == 0) {
      cfg.keys_path = require_value(arg);
    } else if (arg.rfind("--input_csv=", 0) == 0) {
      cfg.input_csv = require_value(arg);
    } else if (arg == "--help") {
      std::cout
          << "Usage synthetic: generate_spatial_zorder --dist=uniform|clustered "
          << "--n=100000 --csv=out.csv --keys=out.keys "
          << "[--clusters=8 --sigma=0.04 --seed=1866]\n";
      std::cout
          << "Usage CSV: generate_spatial_zorder --input_csv=in.csv "
          << "--csv=normalized.csv --keys=out.keys\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }
  if (cfg.csv_path.empty()) {
    throw std::invalid_argument("--csv is required");
  }
  if (cfg.keys_path.empty()) {
    throw std::invalid_argument("--keys is required");
  }
  if (cfg.dist != "uniform" && cfg.dist != "clustered") {
    throw std::invalid_argument("--dist must be uniform or clustered");
  }
  if (cfg.n == 0) {
    throw std::invalid_argument("--n must be positive");
  }
  if (cfg.clusters == 0) {
    throw std::invalid_argument("--clusters must be positive");
  }
  if (cfg.sigma <= 0.0) {
    throw std::invalid_argument("--sigma must be positive");
  }
  return cfg;
}

std::string strip_cr(std::string value) {
  if (!value.empty() && value.back() == '\r') {
    value.pop_back();
  }
  return value;
}

std::uint64_t spread_bits(std::uint32_t value) {
  // 将 32-bit 整数的每一位展开到偶数位：
  // abc... -> a0b0c0...
  // 后续把 x 放偶数位、y 放奇数位，就得到 Morton/Z-order 编码。
  std::uint64_t x = value;
  x = (x | (x << 16)) & 0x0000ffff0000ffffULL;
  x = (x | (x << 8)) & 0x00ff00ff00ff00ffULL;
  x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0fULL;
  x = (x | (x << 2)) & 0x3333333333333333ULL;
  x = (x | (x << 1)) & 0x5555555555555555ULL;
  return x;
}

std::uint32_t quantize_unit(double value) {
  // 输入点坐标在 [0, 1] 内。先裁剪，再映射到 uint32 网格。
  value = std::clamp(value, 0.0, 1.0);
  constexpr double max_u32 = 4294967295.0;
  return static_cast<std::uint32_t>(value * max_u32);
}

std::uint64_t zorder_key(double x, double y) {
  // Z-order/Morton code：
  // x 的 bit 放在 0,2,4,... 位，y 的 bit 放在 1,3,5,... 位。
  // 这样二维点 (x,y) 就变成一个可以交给 GRE 的一维 uint64 key。
  const auto xi = quantize_unit(x);
  const auto yi = quantize_unit(y);
  return spread_bits(xi) | (spread_bits(yi) << 1);
}

void write_outputs(const Config &cfg, const std::vector<std::pair<double, double>> &points) {
  std::ofstream csv(cfg.csv_path);
  if (!csv.is_open()) {
    throw std::runtime_error("Cannot open CSV output: " + cfg.csv_path);
  }
  csv << std::setprecision(17);

  std::ofstream keys(cfg.keys_path, std::ios::binary);
  if (!keys.is_open()) {
    throw std::runtime_error("Cannot open key output: " + cfg.keys_path);
  }

  // GRE 的 binary loader 约定：文件第一个 uint64 是 key 数量，
  // 后面连续写入 n 个 uint64 key。
  const std::uint64_t n = points.size();
  keys.write(reinterpret_cast<const char *>(&n), sizeof(n));

  std::uint64_t duplicate_neighbors = 0;
  std::uint64_t prev_key = 0;
  bool has_prev = false;
  for (const auto &[x, y] : points) {
    const auto key = zorder_key(x, y);
    csv << x << ',' << y << '\n';
    keys.write(reinterpret_cast<const char *>(&key), sizeof(key));
    if (has_prev && key == prev_key) {
      ++duplicate_neighbors;
    }
    prev_key = key;
    has_prev = true;
  }

  std::cout << "Wrote " << points.size() << " 2D points to " << cfg.csv_path << '\n';
  std::cout << "Wrote GRE binary uint64 keys to " << cfg.keys_path << '\n';
  std::cout << "Adjacent duplicate Z-order keys before GRE shuffle/sort: "
            << duplicate_neighbors << '\n';
}

std::vector<std::pair<double, double>> generate_uniform(const Config &cfg) {
  // Uniform 2D：x 和 y 都独立服从 U(0,1)。
  std::mt19937_64 gen(cfg.seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<std::pair<double, double>> points;
  points.reserve(cfg.n);
  for (std::size_t i = 0; i < cfg.n; ++i) {
    points.emplace_back(unit(gen), unit(gen));
  }
  return points;
}

std::vector<std::pair<double, double>> generate_clustered(const Config &cfg) {
  // Clustered 2D：
  // 1. 先在 [0,1]x[0,1] 内随机生成若干簇中心。
  // 2. 每个点随机选择一个簇中心。
  // 3. 在簇中心附近按 Gaussian 分布采样。
  // 4. 最后裁剪回 [0,1]x[0,1]。
  std::mt19937_64 gen(cfg.seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_int_distribution<std::size_t> choose_cluster(0, cfg.clusters - 1);
  std::normal_distribution<double> gaussian(0.0, cfg.sigma);

  std::vector<std::pair<double, double>> centers;
  centers.reserve(cfg.clusters);
  for (std::size_t i = 0; i < cfg.clusters; ++i) {
    centers.emplace_back(unit(gen), unit(gen));
  }

  std::vector<std::pair<double, double>> points;
  points.reserve(cfg.n);
  for (std::size_t i = 0; i < cfg.n; ++i) {
    const auto &[cx, cy] = centers[choose_cluster(gen)];
    points.emplace_back(std::clamp(cx + gaussian(gen), 0.0, 1.0),
                        std::clamp(cy + gaussian(gen), 0.0, 1.0));
  }
  return points;
}

std::vector<std::pair<double, double>> read_and_normalize_csv(const Config &cfg) {
  // 读取已有二维点 CSV。当前按两列解析：X,Y。
  // 原始坐标可以是经纬度或任意实数范围；这里会按 min/max 归一化到 [0,1]。
  std::ifstream input(cfg.input_csv);
  if (!input.is_open()) {
    throw std::runtime_error("Cannot open input CSV: " + cfg.input_csv);
  }

  std::string line;
  std::getline(input, line);  // 跳过表头 X,Y。

  std::vector<std::pair<double, double>> raw_points;
  raw_points.reserve(cfg.n);

  double min_x = 0.0;
  double max_x = 0.0;
  double min_y = 0.0;
  double max_y = 0.0;
  bool first = true;

  while (std::getline(input, line)) {
    line = strip_cr(line);
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string x_text;
    std::string y_text;
    if (!std::getline(ss, x_text, ',') || !std::getline(ss, y_text, ',')) {
      throw std::runtime_error("Bad CSV line: " + line);
    }

    const double x = std::stod(x_text);
    const double y = std::stod(y_text);
    raw_points.emplace_back(x, y);

    if (first) {
      min_x = max_x = x;
      min_y = max_y = y;
      first = false;
    } else {
      min_x = std::min(min_x, x);
      max_x = std::max(max_x, x);
      min_y = std::min(min_y, y);
      max_y = std::max(max_y, y);
    }
  }

  if (raw_points.empty()) {
    throw std::runtime_error("Input CSV has no data points");
  }
  if (min_x == max_x || min_y == max_y) {
    throw std::runtime_error("Input CSV range is degenerate");
  }

  std::vector<std::pair<double, double>> normalized;
  normalized.reserve(raw_points.size());
  for (const auto &[x, y] : raw_points) {
    normalized.emplace_back((x - min_x) / (max_x - min_x),
                            (y - min_y) / (max_y - min_y));
  }

  std::cout << "Read " << raw_points.size() << " points from " << cfg.input_csv << '\n';
  std::cout << "Original bounds: x=[" << std::setprecision(17) << min_x << ", "
            << max_x << "], y=[" << min_y << ", " << max_y << "]\n";
  return normalized;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const auto cfg = parse_args(argc, argv);
    const auto points = !cfg.input_csv.empty()
                            ? read_and_normalize_csv(cfg)
                            : (cfg.dist == "uniform" ? generate_uniform(cfg)
                                                     : generate_clustered(cfg));
    write_outputs(cfg, points);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
