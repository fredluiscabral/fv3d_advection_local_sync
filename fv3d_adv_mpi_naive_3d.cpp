#include "fv3d_adv_common.hpp"

#include <mpi.h>

#include <array>
#include <iostream>
#include <vector>

namespace {

inline std::size_t lidx(int i, int j, int k, int nx, int ny) {
    return (static_cast<std::size_t>(k) * (ny + 2) + static_cast<std::size_t>(j)) * (nx + 2)
           + static_cast<std::size_t>(i);
}

inline int local_begin(int coord, int parts, int N) {
    return static_cast<int>((static_cast<long long>(coord) * N) / parts);
}

inline int local_end(int coord, int parts, int N) {
    return static_cast<int>((static_cast<long long>(coord + 1) * N) / parts);
}

void pack_x(const std::vector<double>& U, std::vector<double>& buf, int fixed_i, int nx, int ny, int nz) {
    std::size_t q = 0;
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            buf[q++] = U[lidx(fixed_i, j, k, nx, ny)];
}

void unpack_x(std::vector<double>& U, const std::vector<double>& buf, int fixed_i, int nx, int ny, int nz) {
    std::size_t q = 0;
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            U[lidx(fixed_i, j, k, nx, ny)] = buf[q++];
}

void pack_y(const std::vector<double>& U, std::vector<double>& buf, int fixed_j, int nx, int ny, int nz) {
    std::size_t q = 0;
    for (int k = 1; k <= nz; ++k)
        for (int i = 1; i <= nx; ++i)
            buf[q++] = U[lidx(i, fixed_j, k, nx, ny)];
}

void unpack_y(std::vector<double>& U, const std::vector<double>& buf, int fixed_j, int nx, int ny, int nz) {
    std::size_t q = 0;
    for (int k = 1; k <= nz; ++k)
        for (int i = 1; i <= nx; ++i)
            U[lidx(i, fixed_j, k, nx, ny)] = buf[q++];
}

void pack_z(const std::vector<double>& U, std::vector<double>& buf, int fixed_k, int nx, int ny, int nz) {
    std::size_t q = 0;
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i)
            buf[q++] = U[lidx(i, j, fixed_k, nx, ny)];
}

void unpack_z(std::vector<double>& U, const std::vector<double>& buf, int fixed_k, int nx, int ny, int nz) {
    std::size_t q = 0;
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i)
            U[lidx(i, j, fixed_k, nx, ny)] = buf[q++];
}

void exchange_halos(std::vector<double>& U,
                    int nx, int ny, int nz,
                    int xm, int xp,
                    int ym, int yp,
                    int zm, int zp,
                    MPI_Comm comm) {
    MPI_Status st;

    std::vector<double> sx(static_cast<std::size_t>(ny) * nz), rx(static_cast<std::size_t>(ny) * nz);
    std::vector<double> sy(static_cast<std::size_t>(nx) * nz), ry(static_cast<std::size_t>(nx) * nz);
    std::vector<double> sz(static_cast<std::size_t>(nx) * ny), rz(static_cast<std::size_t>(nx) * ny);

    // x direction
    pack_x(U, sx, 1, nx, ny, nz);
    MPI_Sendrecv(sx.data(), static_cast<int>(sx.size()), MPI_DOUBLE, xm, 100,
                 rx.data(), static_cast<int>(rx.size()), MPI_DOUBLE, xp, 100,
                 comm, &st);
    unpack_x(U, rx, nx + 1, nx, ny, nz);

    pack_x(U, sx, nx, nx, ny, nz);
    MPI_Sendrecv(sx.data(), static_cast<int>(sx.size()), MPI_DOUBLE, xp, 101,
                 rx.data(), static_cast<int>(rx.size()), MPI_DOUBLE, xm, 101,
                 comm, &st);
    unpack_x(U, rx, 0, nx, ny, nz);

    // y direction
    pack_y(U, sy, 1, nx, ny, nz);
    MPI_Sendrecv(sy.data(), static_cast<int>(sy.size()), MPI_DOUBLE, ym, 200,
                 ry.data(), static_cast<int>(ry.size()), MPI_DOUBLE, yp, 200,
                 comm, &st);
    unpack_y(U, ry, ny + 1, nx, ny, nz);

    pack_y(U, sy, ny, nx, ny, nz);
    MPI_Sendrecv(sy.data(), static_cast<int>(sy.size()), MPI_DOUBLE, yp, 201,
                 ry.data(), static_cast<int>(ry.size()), MPI_DOUBLE, ym, 201,
                 comm, &st);
    unpack_y(U, ry, 0, nx, ny, nz);

    // z direction
    pack_z(U, sz, 1, nx, ny, nz);
    MPI_Sendrecv(sz.data(), static_cast<int>(sz.size()), MPI_DOUBLE, zm, 300,
                 rz.data(), static_cast<int>(rz.size()), MPI_DOUBLE, zp, 300,
                 comm, &st);
    unpack_z(U, rz, nz + 1, nx, ny, nz);

    pack_z(U, sz, nz, nx, ny, nz);
    MPI_Sendrecv(sz.data(), static_cast<int>(sz.size()), MPI_DOUBLE, zp, 301,
                 rz.data(), static_cast<int>(rz.size()), MPI_DOUBLE, zm, 301,
                 comm, &st);
    unpack_z(U, rz, 0, nx, ny, nz);
}

void update_local(const std::vector<double>& src,
                  std::vector<double>& dst,
                  int nx, int ny, int nz,
                  const fv3d::Params& p,
                  double dt) {
    const double dx = 1.0 / static_cast<double>(p.N);
    const double cx = p.a * dt / dx;
    const double cy = p.b * dt / dx;
    const double cz = p.c * dt / dx;

    for (int k = 1; k <= nz; ++k) {
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                const double u = src[lidx(i, j, k, nx, ny)];
                double out = u;

                if (p.a >= 0.0) {
                    out -= cx * (u - src[lidx(i - 1, j, k, nx, ny)]);
                } else {
                    out -= cx * (src[lidx(i + 1, j, k, nx, ny)] - u);
                }

                if (p.b >= 0.0) {
                    out -= cy * (u - src[lidx(i, j - 1, k, nx, ny)]);
                } else {
                    out -= cy * (src[lidx(i, j + 1, k, nx, ny)] - u);
                }

                if (p.c >= 0.0) {
                    out -= cz * (u - src[lidx(i, j, k - 1, nx, ny)]);
                } else {
                    out -= cz * (src[lidx(i, j, k + 1, nx, ny)] - u);
                }

                dst[lidx(i, j, k, nx, ny)] = out;
            }
        }
    }
}

} // anonymous namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const fv3d::Params p = fv3d::read_params();

    int dims[3] = {0, 0, 0};
    MPI_Dims_create(size, 3, dims);

    int periods[3] = {1, 1, 1};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);

    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);

    int xm, xp, ym, yp, zm, zp;
    MPI_Cart_shift(cart, 0, 1, &xm, &xp);
    MPI_Cart_shift(cart, 1, 1, &ym, &yp);
    MPI_Cart_shift(cart, 2, 1, &zm, &zp);

    const int xs = local_begin(coords[0], dims[0], p.N);
    const int xe = local_end(coords[0], dims[0], p.N);
    const int ys = local_begin(coords[1], dims[1], p.N);
    const int ye = local_end(coords[1], dims[1], p.N);
    const int zs = local_begin(coords[2], dims[2], p.N);
    const int ze = local_end(coords[2], dims[2], p.N);

    const int nx = xe - xs;
    const int ny = ye - ys;
    const int nz = ze - zs;

    const std::size_t local_size =
        static_cast<std::size_t>(nx + 2) * static_cast<std::size_t>(ny + 2) * static_cast<std::size_t>(nz + 2);

    std::vector<double> U0(local_size, 0.0);
    std::vector<double> U1(local_size, 0.0);

    for (int k = 1; k <= nz; ++k) {
        const int gk = zs + (k - 1);
        const double z = (static_cast<double>(gk) + 0.5) / static_cast<double>(p.N);
        for (int j = 1; j <= ny; ++j) {
            const int gj = ys + (j - 1);
            const double y = (static_cast<double>(gj) + 0.5) / static_cast<double>(p.N);
            for (int i = 1; i <= nx; ++i) {
                const int gi = xs + (i - 1);
                const double x = (static_cast<double>(gi) + 0.5) / static_cast<double>(p.N);
                U0[lidx(i, j, k, nx, ny)] =
                    fv3d::periodic_gaussian(x, y, z, p.x0, p.y0, p.z0, p.D);
            }
        }
    }

    const double dt = fv3d::compute_dt(p);

    MPI_Barrier(cart);
    const double t0 = MPI_Wtime();

    for (int step = 0; step < p.steps; ++step) {
        std::vector<double>& src = (step % 2 == 0) ? U0 : U1;
        std::vector<double>& dst = (step % 2 == 0) ? U1 : U0;

        exchange_halos(src, nx, ny, nz, xm, xp, ym, yp, zm, zp, cart);
        update_local(src, dst, nx, ny, nz, p, dt);
    }

    MPI_Barrier(cart);
    const double elapsed = MPI_Wtime() - t0;

    const std::vector<double>& finalU = (p.steps % 2 == 0) ? U0 : U1;
    const double final_time = dt * static_cast<double>(p.steps);

    const double xc = fv3d::wrap01(p.x0 + p.a * final_time);
    const double yc = fv3d::wrap01(p.y0 + p.b * final_time);
    const double zc = fv3d::wrap01(p.z0 + p.c * final_time);

    double l1_local = 0.0;
    double l2_local = 0.0;
    double linf_local = 0.0;

    for (int k = 1; k <= nz; ++k) {
        const int gk = zs + (k - 1);
        const double z = (static_cast<double>(gk) + 0.5) / static_cast<double>(p.N);
        for (int j = 1; j <= ny; ++j) {
            const int gj = ys + (j - 1);
            const double y = (static_cast<double>(gj) + 0.5) / static_cast<double>(p.N);
            for (int i = 1; i <= nx; ++i) {
                const int gi = xs + (i - 1);
                const double x = (static_cast<double>(gi) + 0.5) / static_cast<double>(p.N);
                const double exact = fv3d::periodic_gaussian(x, y, z, xc, yc, zc, p.D);
                const double e = std::fabs(finalU[lidx(i, j, k, nx, ny)] - exact);
                l1_local += e;
                l2_local += e * e;
                linf_local = std::max(linf_local, e);
            }
        }
    }

    double l1_global = 0.0;
    double l2_global = 0.0;
    double linf_global = 0.0;

    MPI_Reduce(&l1_local, &l1_global, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
    MPI_Reduce(&l2_local, &l2_global, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
    MPI_Reduce(&linf_local, &linf_global, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    if (rank == 0) {
        const double total = static_cast<double>(p.N) * p.N * p.N;
        fv3d::ErrorMetrics err;
        err.l1 = l1_global / total;
        err.l2 = std::sqrt(l2_global / total);
        err.linf = linf_global;

        fv3d::print_report("mpi_puro", p, size, dims[0], dims[1], dims[2], dt, elapsed, err);
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
