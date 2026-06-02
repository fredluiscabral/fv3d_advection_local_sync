#ifndef FV3D_ADV_DIAG_COMMON_HPP
#define FV3D_ADV_DIAG_COMMON_HPP

#include "fv3d_adv_common.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace fv3d_diag {

class AlignedDoubles {
public:
    AlignedDoubles() = default;
    AlignedDoubles(const AlignedDoubles&) = delete;
    AlignedDoubles& operator=(const AlignedDoubles&) = delete;

    AlignedDoubles(AlignedDoubles&& other) noexcept
        : ptr_(other.ptr_), n_(other.n_) {
        other.ptr_ = nullptr;
        other.n_ = 0;
    }

    AlignedDoubles& operator=(AlignedDoubles&& other) noexcept {
        if (this != &other) {
            std::free(ptr_);
            ptr_ = other.ptr_;
            n_ = other.n_;
            other.ptr_ = nullptr;
            other.n_ = 0;
        }
        return *this;
    }

    ~AlignedDoubles() { std::free(ptr_); }

    bool allocate(std::size_t n, std::size_t alignment = 64) {
        std::free(ptr_);
        ptr_ = nullptr;
        n_ = 0;
        if (n == 0) return true;
        void* raw = nullptr;
        const std::size_t bytes = n * sizeof(double);
        if (posix_memalign(&raw, alignment, bytes) != 0 || raw == nullptr) {
            return false;
        }
        ptr_ = static_cast<double*>(raw);
        n_ = n;
        return true;
    }

    double* data() noexcept { return ptr_; }
    const double* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return n_; }

private:
    double* ptr_ = nullptr;
    std::size_t n_ = 0;
};

inline void initialize_global_block(double* U0,
                                    double* U1,
                                    const fv3d::Block& b,
                                    const fv3d::Params& p) {
    const int N = p.N;
    for (int k = b.z0; k < b.z1; ++k) {
        const double z = (static_cast<double>(k) + 0.5) / static_cast<double>(N);
        for (int j = b.y0; j < b.y1; ++j) {
            const double y = (static_cast<double>(j) + 0.5) / static_cast<double>(N);
            for (int i = b.x0; i < b.x1; ++i) {
                const double x = (static_cast<double>(i) + 0.5) / static_cast<double>(N);
                const std::size_t q = fv3d::idx3(i, j, k, N);
                U0[q] = fv3d::periodic_gaussian(x, y, z, p.x0, p.y0, p.z0, p.D);
                U1[q] = 0.0;
            }
        }
    }
}

inline void update_global_block_raw(const double* src,
                                    double* dst,
                                    const fv3d::Block& b,
                                    const fv3d::Params& p,
                                    double dt) {
    const int N = p.N;
    const int tile = p.tile;
    const double dx = 1.0 / static_cast<double>(N);
    const double cx = p.a * dt / dx;
    const double cy = p.b * dt / dx;
    const double cz = p.c * dt / dx;

    for (int kk = b.z0; kk < b.z1; kk += tile) {
        const int kend = std::min(b.z1, kk + tile);
        for (int jj = b.y0; jj < b.y1; jj += tile) {
            const int jend = std::min(b.y1, jj + tile);
            for (int ii = b.x0; ii < b.x1; ii += tile) {
                const int iend = std::min(b.x1, ii + tile);

                for (int k = kk; k < kend; ++k) {
                    const int km = (k == 0) ? N - 1 : k - 1;
                    const int kp = (k == N - 1) ? 0 : k + 1;
                    for (int j = jj; j < jend; ++j) {
                        const int jm = (j == 0) ? N - 1 : j - 1;
                        const int jp = (j == N - 1) ? 0 : j + 1;
                        for (int i = ii; i < iend; ++i) {
                            const int im = (i == 0) ? N - 1 : i - 1;
                            const int ip = (i == N - 1) ? 0 : i + 1;

                            const std::size_t center = fv3d::idx3(i, j, k, N);
                            const double u = src[center];
                            double outv = u;

                            if (p.a >= 0.0) {
                                outv -= cx * (u - src[fv3d::idx3(im, j, k, N)]);
                            } else {
                                outv -= cx * (src[fv3d::idx3(ip, j, k, N)] - u);
                            }

                            if (p.b >= 0.0) {
                                outv -= cy * (u - src[fv3d::idx3(i, jm, k, N)]);
                            } else {
                                outv -= cy * (src[fv3d::idx3(i, jp, k, N)] - u);
                            }

                            if (p.c >= 0.0) {
                                outv -= cz * (u - src[fv3d::idx3(i, j, km, N)]);
                            } else {
                                outv -= cz * (src[fv3d::idx3(i, j, kp, N)] - u);
                            }

                            dst[center] = outv;
                        }
                    }
                }
            }
        }
    }
}

inline fv3d::ErrorMetrics compute_errors_raw(const double* U,
                                             const fv3d::Params& p,
                                             double final_time) {
    const int N = p.N;
    const double xc = fv3d::wrap01(p.x0 + p.a * final_time);
    const double yc = fv3d::wrap01(p.y0 + p.b * final_time);
    const double zc = fv3d::wrap01(p.z0 + p.c * final_time);

    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;

    #pragma omp parallel for collapse(3) reduction(+:l1,l2) reduction(max:linf) schedule(static)
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                const double x = (static_cast<double>(i) + 0.5) / static_cast<double>(N);
                const double y = (static_cast<double>(j) + 0.5) / static_cast<double>(N);
                const double z = (static_cast<double>(k) + 0.5) / static_cast<double>(N);
                const double exact = fv3d::periodic_gaussian(x, y, z, xc, yc, zc, p.D);
                const double e = std::fabs(U[fv3d::idx3(i, j, k, N)] - exact);
                l1 += e;
                l2 += e * e;
                linf = std::max(linf, e);
            }
        }
    }

    const double total = static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    fv3d::ErrorMetrics err;
    err.l1 = l1 / total;
    err.l2 = std::sqrt(l2 / total);
    err.linf = linf;
    return err;
}

inline void print_zero_error_report(const char* variant,
                                    const fv3d::Params& p,
                                    int workers,
                                    int px,
                                    int py,
                                    int pz,
                                    double dt,
                                    double elapsed) {
    fv3d::ErrorMetrics err;
    err.l1 = 0.0;
    err.l2 = 0.0;
    err.linf = 0.0;
    fv3d::print_report(variant, p, workers, px, py, pz, dt, elapsed, err);
}

} // namespace fv3d_diag

#endif // FV3D_ADV_DIAG_COMMON_HPP
