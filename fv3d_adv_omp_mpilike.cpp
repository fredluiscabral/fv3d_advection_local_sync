#include "fv3d_adv_common.hpp"

#include <iostream>
#include <memory>
#include <vector>

namespace {

inline std::size_t lidx(int i, int j, int k, int nx, int ny) {
    return (static_cast<std::size_t>(k) * static_cast<std::size_t>(ny + 2)
            + static_cast<std::size_t>(j)) * static_cast<std::size_t>(nx + 2)
           + static_cast<std::size_t>(i);
}

struct LocalBlock3D {
    int xs = 0, xe = 0;
    int ys = 0, ye = 0;
    int zs = 0, ze = 0;
    int nx = 0, ny = 0, nz = 0;
    int bx = 0, by = 0, bz = 0;
    std::vector<double> U0;
    std::vector<double> U1;
};

inline int neighbor_id(int bx, int by, int bz, int px, int py) {
    return (bz * py + by) * px + bx;
}

inline void init_local(LocalBlock3D& b, const fv3d::Params& p) {
    const std::size_t local_size = static_cast<std::size_t>(b.nx + 2) * (b.ny + 2) * (b.nz + 2);
    b.U0.assign(local_size, 0.0);
    b.U1.assign(local_size, 0.0);

    for (int k = 1; k <= b.nz; ++k) {
        const int gk = b.zs + (k - 1);
        const double z = (static_cast<double>(gk) + 0.5) / static_cast<double>(p.N);
        for (int j = 1; j <= b.ny; ++j) {
            const int gj = b.ys + (j - 1);
            const double y = (static_cast<double>(gj) + 0.5) / static_cast<double>(p.N);
            for (int i = 1; i <= b.nx; ++i) {
                const int gi = b.xs + (i - 1);
                const double x = (static_cast<double>(gi) + 0.5) / static_cast<double>(p.N);
                b.U0[lidx(i, j, k, b.nx, b.ny)] =
                    fv3d::periodic_gaussian(x, y, z, p.x0, p.y0, p.z0, p.D);
            }
        }
    }
}

inline const double* src_ptr(const LocalBlock3D& b, int step) {
    return (step % 2 == 0) ? b.U0.data() : b.U1.data();
}

inline double* src_ptr_mut(LocalBlock3D& b, int step) {
    return (step % 2 == 0) ? b.U0.data() : b.U1.data();
}

inline double* dst_ptr(LocalBlock3D& b, int step) {
    return (step % 2 == 0) ? b.U1.data() : b.U0.data();
}

inline void copy_xminus_halo(LocalBlock3D& me, const LocalBlock3D& nb, int step) {
    double* dst = src_ptr_mut(me, step);
    const double* src = src_ptr(nb, step);
    for (int k = 1; k <= me.nz; ++k) {
        for (int j = 1; j <= me.ny; ++j) {
            dst[lidx(0, j, k, me.nx, me.ny)] = src[lidx(nb.nx, j, k, nb.nx, nb.ny)];
        }
    }
}

inline void copy_xplus_halo(LocalBlock3D& me, const LocalBlock3D& nb, int step) {
    double* dst = src_ptr_mut(me, step);
    const double* src = src_ptr(nb, step);
    for (int k = 1; k <= me.nz; ++k) {
        for (int j = 1; j <= me.ny; ++j) {
            dst[lidx(me.nx + 1, j, k, me.nx, me.ny)] = src[lidx(1, j, k, nb.nx, nb.ny)];
        }
    }
}

inline void copy_yminus_halo(LocalBlock3D& me, const LocalBlock3D& nb, int step) {
    double* dst = src_ptr_mut(me, step);
    const double* src = src_ptr(nb, step);
    for (int k = 1; k <= me.nz; ++k) {
        for (int i = 1; i <= me.nx; ++i) {
            dst[lidx(i, 0, k, me.nx, me.ny)] = src[lidx(i, nb.ny, k, nb.nx, nb.ny)];
        }
    }
}

inline void copy_yplus_halo(LocalBlock3D& me, const LocalBlock3D& nb, int step) {
    double* dst = src_ptr_mut(me, step);
    const double* src = src_ptr(nb, step);
    for (int k = 1; k <= me.nz; ++k) {
        for (int i = 1; i <= me.nx; ++i) {
            dst[lidx(i, me.ny + 1, k, me.nx, me.ny)] = src[lidx(i, 1, k, nb.nx, nb.ny)];
        }
    }
}

inline void copy_zminus_halo(LocalBlock3D& me, const LocalBlock3D& nb, int step) {
    double* dst = src_ptr_mut(me, step);
    const double* src = src_ptr(nb, step);
    for (int j = 1; j <= me.ny; ++j) {
        for (int i = 1; i <= me.nx; ++i) {
            dst[lidx(i, j, 0, me.nx, me.ny)] = src[lidx(i, j, nb.nz, nb.nx, nb.ny)];
        }
    }
}

inline void copy_zplus_halo(LocalBlock3D& me, const LocalBlock3D& nb, int step) {
    double* dst = src_ptr_mut(me, step);
    const double* src = src_ptr(nb, step);
    for (int j = 1; j <= me.ny; ++j) {
        for (int i = 1; i <= me.nx; ++i) {
            dst[lidx(i, j, me.nz + 1, me.nx, me.ny)] = src[lidx(i, j, 1, nb.nx, nb.ny)];
        }
    }
}

inline void exchange_halos_shared(std::vector<LocalBlock3D>& blocks,
                                  const fv3d::Decomp& d,
                                  int tid,
                                  int step) {
    LocalBlock3D& me = blocks[tid];

    const int xm_bx = (me.bx - 1 + d.px) % d.px;
    const int xp_bx = (me.bx + 1) % d.px;
    const int ym_by = (me.by - 1 + d.py) % d.py;
    const int yp_by = (me.by + 1) % d.py;
    const int zm_bz = (me.bz - 1 + d.pz) % d.pz;
    const int zp_bz = (me.bz + 1) % d.pz;

    const int xm = neighbor_id(xm_bx, me.by, me.bz, d.px, d.py);
    const int xp = neighbor_id(xp_bx, me.by, me.bz, d.px, d.py);
    const int ym = neighbor_id(me.bx, ym_by, me.bz, d.px, d.py);
    const int yp = neighbor_id(me.bx, yp_by, me.bz, d.px, d.py);
    const int zm = neighbor_id(me.bx, me.by, zm_bz, d.px, d.py);
    const int zp = neighbor_id(me.bx, me.by, zp_bz, d.px, d.py);

    copy_xminus_halo(me, blocks[xm], step);
    copy_xplus_halo(me, blocks[xp], step);
    copy_yminus_halo(me, blocks[ym], step);
    copy_yplus_halo(me, blocks[yp], step);
    copy_zminus_halo(me, blocks[zm], step);
    copy_zplus_halo(me, blocks[zp], step);
}

inline void update_local(const LocalBlock3D& b,
                         const double* src,
                         double* dst,
                         const fv3d::Params& p,
                         double dt) {
    const double dx = 1.0 / static_cast<double>(p.N);
    const double cx = p.a * dt / dx;
    const double cy = p.b * dt / dx;
    const double cz = p.c * dt / dx;
    const int tile = p.tile;

    for (int kk = 1; kk <= b.nz; kk += tile) {
        const int kend = std::min(b.nz + 1, kk + tile);
        for (int jj = 1; jj <= b.ny; jj += tile) {
            const int jend = std::min(b.ny + 1, jj + tile);
            for (int ii = 1; ii <= b.nx; ii += tile) {
                const int iend = std::min(b.nx + 1, ii + tile);

                for (int k = kk; k < kend; ++k) {
                    for (int j = jj; j < jend; ++j) {
                        for (int i = ii; i < iend; ++i) {
                            const double u = src[lidx(i, j, k, b.nx, b.ny)];
                            double out = u;

                            if (p.a >= 0.0) {
                                out -= cx * (u - src[lidx(i - 1, j, k, b.nx, b.ny)]);
                            } else {
                                out -= cx * (src[lidx(i + 1, j, k, b.nx, b.ny)] - u);
                            }

                            if (p.b >= 0.0) {
                                out -= cy * (u - src[lidx(i, j - 1, k, b.nx, b.ny)]);
                            } else {
                                out -= cy * (src[lidx(i, j + 1, k, b.nx, b.ny)] - u);
                            }

                            if (p.c >= 0.0) {
                                out -= cz * (u - src[lidx(i, j, k - 1, b.nx, b.ny)]);
                            } else {
                                out -= cz * (src[lidx(i, j, k + 1, b.nx, b.ny)] - u);
                            }

                            dst[lidx(i, j, k, b.nx, b.ny)] = out;
                        }
                    }
                }
            }
        }
    }
}

inline fv3d::ErrorMetrics compute_local_errors(const std::vector<LocalBlock3D>& blocks,
                                               const fv3d::Params& p,
                                               double final_time) {
    const double xc = fv3d::wrap01(p.x0 + p.a * final_time);
    const double yc = fv3d::wrap01(p.y0 + p.b * final_time);
    const double zc = fv3d::wrap01(p.z0 + p.c * final_time);

    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
    const int P = static_cast<int>(blocks.size());

    #pragma omp parallel for reduction(+:l1,l2) reduction(max:linf) schedule(static)
    for (int tid = 0; tid < P; ++tid) {
        const LocalBlock3D& b = blocks[tid];
        const double* U = (p.steps % 2 == 0) ? b.U0.data() : b.U1.data();

        for (int k = 1; k <= b.nz; ++k) {
            const int gk = b.zs + (k - 1);
            const double z = (static_cast<double>(gk) + 0.5) / static_cast<double>(p.N);
            for (int j = 1; j <= b.ny; ++j) {
                const int gj = b.ys + (j - 1);
                const double y = (static_cast<double>(gj) + 0.5) / static_cast<double>(p.N);
                for (int i = 1; i <= b.nx; ++i) {
                    const int gi = b.xs + (i - 1);
                    const double x = (static_cast<double>(gi) + 0.5) / static_cast<double>(p.N);
                    const double exact = fv3d::periodic_gaussian(x, y, z, xc, yc, zc, p.D);
                    const double e = std::fabs(U[lidx(i, j, k, b.nx, b.ny)] - exact);
                    l1 += e;
                    l2 += e * e;
                    linf = std::max(linf, e);
                }
            }
        }
    }

    const double total = static_cast<double>(p.N) * p.N * p.N;
    fv3d::ErrorMetrics err;
    err.l1 = l1 / total;
    err.l2 = std::sqrt(l2 / total);
    err.linf = linf;
    return err;
}

} // anonymous namespace

int main() {
    const fv3d::Params p = fv3d::read_params();
    const double dt = fv3d::compute_dt(p);
    const int P = omp_get_max_threads();
    const fv3d::Decomp decomp = fv3d::build_decomp(P, p.N);

    std::vector<LocalBlock3D> blocks(P);

    for (int tid = 0; tid < P; ++tid) {
        int bx, by, bz;
        fv3d::id_to_coord(tid, decomp.px, decomp.py, bx, by, bz);
        const fv3d::Block gb = decomp.blocks[tid];
        LocalBlock3D& b = blocks[tid];
        b.bx = bx; b.by = by; b.bz = bz;
        b.xs = gb.x0; b.xe = gb.x1;
        b.ys = gb.y0; b.ye = gb.y1;
        b.zs = gb.z0; b.ze = gb.z1;
        b.nx = b.xe - b.xs;
        b.ny = b.ye - b.ys;
        b.nz = b.ze - b.zs;
    }

    // First-touch local: cada thread aloca/inicializa seu próprio subdomínio.
    #pragma omp parallel num_threads(P)
    {
        const int tid = omp_get_thread_num();
        init_local(blocks[tid], p);
    }

    std::vector<fv3d::PaddedAtomicInt> progress(P);
    for (int i = 0; i < P; ++i) fv3d::store_progress(progress[i], 0);

    const double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(P)
    {
        const int tid = omp_get_thread_num();
        LocalBlock3D& b = blocks[tid];
        const auto& neighbors = decomp.neighbors[tid];

        for (int step = 0; step < p.steps; ++step) {
            fv3d::wait_neighbors_progress(progress, neighbors, step);

            exchange_halos_shared(blocks, decomp, tid, step);

            const double* src = src_ptr(b, step);
            double* dst = dst_ptr(b, step);
            update_local(b, src, dst, p, dt);

            fv3d::store_progress(progress[tid], step + 1);
        }
    }

    const double elapsed = omp_get_wtime() - t0;
    const fv3d::ErrorMetrics err = compute_local_errors(blocks, p, dt * static_cast<double>(p.steps));
    fv3d::print_report("omp_mpilike_3d", p, P, decomp.px, decomp.py, decomp.pz, dt, elapsed, err);
    return 0;
}
