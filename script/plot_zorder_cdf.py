import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    ("Uniform", "data/spatial/uniform_2d_1M.csv"),
    ("Clustered", "data/spatial/clustered_2d_1M.csv"),
    ("ENC dedup", "data/spatial/enc_dedup_2d_normalized_1M.csv"),
]

OUT_DIR = "results/zorder_cdf"
MAX_U32 = np.float64(2**32 - 1)
MAX_U64 = np.float64(2**64 - 1)


def spread_bits_u32(values):
    """把 uint32 的 bit 展开到偶数位：abc -> a0b0c0。"""
    x = values.astype(np.uint64)
    x = (x | (x << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x | (x << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x << np.uint64(2))) & np.uint64(0x3333333333333333)
    x = (x | (x << np.uint64(1))) & np.uint64(0x5555555555555555)
    return x


def zorder_keys(points):
    """把二维点 x,y 量化到 uint32 网格，再交错成 uint64 Z-order key。"""
    xy = np.clip(points, 0.0, 1.0)
    xi = (xy[:, 0] * MAX_U32).astype(np.uint32)
    yi = (xy[:, 1] * MAX_U32).astype(np.uint32)
    return spread_bits_u32(xi) | (spread_bits_u32(yi) << np.uint64(1))


def load_cdf(path):
    """读取 x,y CSV，计算 sorted Z-order key 和 rank CDF。"""
    points = np.loadtxt(path, delimiter=",", dtype=np.float64)
    keys = np.sort(zorder_keys(points))
    x = keys.astype(np.float64) / MAX_U64
    y = (np.arange(1, keys.size + 1, dtype=np.float64)) / keys.size
    return x, y


def plot_combined(curves):
    plt.figure(figsize=(8, 5))
    for name, x, y in curves:
        plt.plot(x, y, linewidth=1.1, label=name)
    plt.xlabel("Normalized Z-order key")
    plt.ylabel("Rank / N")
    plt.title("CDF after Z-order Linearization")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "zorder_cdf_combined.png"), dpi=240)
    plt.close()


def plot_single(name, x, y):
    safe_name = name.lower().replace(" ", "_")
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linewidth=1.0)
    plt.xlabel("Normalized Z-order key")
    plt.ylabel("Rank / N")
    plt.title(f"{name}: CDF after Z-order Linearization")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"zorder_cdf_{safe_name}.png"), dpi=240)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    curves = []
    for name, path in DATASETS:
        print(f"Loading and encoding {name}: {path}")
        x, y = load_cdf(path)
        curves.append((name, x, y))
        plot_single(name, x, y)
    plot_combined(curves)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
