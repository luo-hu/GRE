#ifndef __LIPPHYBRID_BASE_H__
#define __LIPPHYBRID_BASE_H__

#include <limits>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstdint>

// Hybrid-LIPP 的节点模型。
// 原版 LIPP 只使用 a * key + b。这里增加一个单调核函数 phi(key)，
// 预测变为 a * phi(key) + b。核函数必须单调，否则 range scan 的顺序假设会被破坏。
template <class T>
class HybridLinearModel
{
public:
    enum Kernel : uint8_t {
        IDENTITY = 0,
        LOG1P = 1,
        SQRT = 2,
        CBRT = 3
    };

    double a = 0; // slope
    double b = 0; // intercept。这里用 double 保持模型结构紧凑，避免每个节点额外膨胀 16 字节。
    Kernel kernel = IDENTITY;
    T origin = 0; // 非线性核函数以当前节点最小 key 为原点，避免负数输入。

    HybridLinearModel() = default;
    HybridLinearModel(double a, double b) : a(a), b(b) {}
    explicit HybridLinearModel(const HybridLinearModel &other)
        : a(other.a), b(other.b), kernel(other.kernel), origin(other.origin) {}

    inline void set_kernel(Kernel new_kernel, T min_key)
    {
        kernel = new_kernel;
        origin = min_key;
    }

    inline long double transform_value(long double raw_key) const
    {
        if (kernel == IDENTITY) {
            return raw_key;
        }

        long double x = raw_key - static_cast<long double>(origin);
        if (x < 0) {
            x = 0;
        }

        switch (kernel) {
        case LOG1P:
            return std::log1pl(x);
        case SQRT:
            return std::sqrt(x);
        case CBRT:
            return std::cbrt(x);
        case IDENTITY:
        default:
            return raw_key;
        }
    }

    inline long double transform(T key) const
    {
        return transform_value(static_cast<long double>(key));
    }

    inline int predict(T key) const
    {
        return std::floor(a * transform(key) + b);
    }

    inline double predict_double(T key) const
    {
        return a * transform(key) + b;
    }
};

#endif // __LIPPHYBRID_BASE_H__
