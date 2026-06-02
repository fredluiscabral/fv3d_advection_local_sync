#ifndef FV3D_ADV_COMMON_HPP
#define FV3D_ADV_COMMON_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <semaphore.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace fv3d {

struct Params {
    int N = 128;
    int steps = 20;
    int tile = 16;

    double a = 1.0;
    double b = 0.5;
    double c = 0.25;
    double cfl = 0.90;

    double D = 100.0;
    double x0 = 0.25;
    double y0 = 0.25;
    double z0 = 0.25;
};

inline std::string trim_copy(std::string s) {
    const char* ws = " \t\r\n";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

inline Params read_params(const std::string& path = "param.txt") {
    Params p;

    std::ifstream in(path);
    if (!in) {
        std::cerr << "ERRO: nao consegui abrir " << path << "\n";
        std::exit(1);
    }

    std::string line;
    while (std::getline(in, line)) {
        const auto pos_hash = line.find('#');
        if (pos_hash != std::string::npos) line = line.substr(0, pos_hash);
        std::replace(line.begin(), line.end(), '=', ' ');
        line = trim_copy(line);
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string key;
        double value = 0.0;
        iss >> key >> value;
        if (!iss) continue;

        if (key == "N") p.N = static_cast<int>(value);
        else if (key == "T" || key == "STEPS" || key == "steps") p.steps = static_cast<int>(value);
        else if (key == "TILE" || key == "tile") p.tile = static_cast<int>(value);
        else if (key == "a") p.a = value;
        else if (key == "b") p.b = value;
        else if (key == "c") p.c = value;
        else if (key == "CFL" || key == "cfl") p.cfl = value;
        else if (key == "D") p.D = value;
        else if (key == "x0") p.x0 = value;
        else if (key == "y0") p.y0 = value;
        else if (key == "z0") p.z0 = value;
    }

    if (p.N <= 2) throw std::runtime_error("N precisa ser maior que 2.");
    if (p.steps < 0) throw std::runtime_error("T/STEPS nao pode ser negativo.");
    if (p.tile <= 0) throw std::runtime_error("TILE precisa ser positivo.");
    if (p.cfl <= 0.0) throw std::runtime_error("CFL precisa ser positivo.");

    return p;
}

inline double wrap01(double x) {
    x -= std::floor(x);
    if (x >= 1.0) x -= 1.0;
    if (x < 0.0) x += 1.0;
    return x;
}

inline double periodic_delta(double x, double xc) {
    double d = std::fabs(x - xc);
    return std::min(d, 1.0 - d);
}

inline double periodic_gaussian(double x, double y, double z,
                                double xc, double yc, double zc, double D) {
    const double dx = periodic_delta(x, xc);
    const double dy = periodic_delta(y, yc);
    const double dz = periodic_delta(z, zc);
    return std::exp(-D * (dx * dx + dy * dy + dz * dz));
}

inline std::size_t idx3(int i, int j, int k, int N) {
    return (static_cast<std::size_t>(k) * N + static_cast<std::size_t>(j)) * N
           + static_cast<std::size_t>(i);
}

inline double compute_dt(const Params& p) {
    const double dx = 1.0 / static_cast<double>(p.N);
    const double denom = std::fabs(p.a) / dx + std::fabs(p.b) / dx + std::fabs(p.c) / dx;
    if (denom <= 0.0) throw std::runtime_error("Velocidades a=b=c=0 nao geram CFL valido.");
    return p.cfl / denom;
}

inline void initialize(std::vector<double>& U, const Params& p) {
    const int N = p.N;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                const double x = (static_cast<double>(i) + 0.5) / static_cast<double>(N);
                const double y = (static_cast<double>(j) + 0.5) / static_cast<double>(N);
                const double z = (static_cast<double>(k) + 0.5) / static_cast<double>(N);
                U[idx3(i, j, k, N)] = periodic_gaussian(x, y, z, p.x0, p.y0, p.z0, p.D);
            }
        }
    }
}

struct Block {
    int x0, x1;
    int y0, y1;
    int z0, z1;
};

struct Decomp {
    int px = 1;
    int py = 1;
    int pz = 1;
    std::vector<Block> blocks;
    std::vector<std::vector<int>> neighbors;
};

inline int block_id(int bx, int by, int bz, int px, int py) {
    return (bz * py + by) * px + bx;
}

inline void id_to_coord(int id, int px, int py, int& bx, int& by, int& bz) {
    bx = id % px;
    const int q = id / px;
    by = q % py;
    bz = q / py;
}

inline int part_begin(int coord, int parts, int N) {
    return static_cast<int>((static_cast<long long>(coord) * N) / parts);
}

inline int part_end(int coord, int parts, int N) {
    return static_cast<int>((static_cast<long long>(coord + 1) * N) / parts);
}

inline void choose_3d_factorization(int P, int& px, int& py, int& pz) {
    px = P; py = 1; pz = 1;
    double best = std::numeric_limits<double>::infinity();

    for (int a = 1; a <= P; ++a) {
        if (P % a != 0) continue;
        const int rem = P / a;
        for (int b = 1; b <= rem; ++b) {
            if (rem % b != 0) continue;
            const int c = rem / b;

            const int dims[3] = {a, b, c};
            const int dmin = std::min({dims[0], dims[1], dims[2]});
            const int dmax = std::max({dims[0], dims[1], dims[2]});

            const double cube_score = static_cast<double>(dmax) / static_cast<double>(dmin);
            const double surface_proxy = static_cast<double>(a + b + c);
            const double score = 1000.0 * cube_score + surface_proxy;

            if (score < best) {
                best = score;
                px = a; py = b; pz = c;
            }
        }
    }

    // Keep x as the fastest-decomposed direction when possible.
    int dims[3] = {px, py, pz};
    std::sort(dims, dims + 3, std::greater<int>());
    px = dims[0];
    py = dims[1];
    pz = dims[2];
}

inline Decomp build_decomp(int P, int N) {
    Decomp d;
    choose_3d_factorization(P, d.px, d.py, d.pz);

    if (d.px * d.py * d.pz != P) {
        throw std::runtime_error("Falha na fatoracao 3D dos workers.");
    }

    d.blocks.resize(P);
    d.neighbors.resize(P);

    for (int id = 0; id < P; ++id) {
        int bx, by, bz;
        id_to_coord(id, d.px, d.py, bx, by, bz);

        d.blocks[id] = {
            part_begin(bx, d.px, N), part_end(bx, d.px, N),
            part_begin(by, d.py, N), part_end(by, d.py, N),
            part_begin(bz, d.pz, N), part_end(bz, d.pz, N)
        };

        auto add_unique = [&](int nx, int ny, int nz) {
            const int nid = block_id(nx, ny, nz, d.px, d.py);
            if (nid == id) return;
            auto& v = d.neighbors[id];
            if (std::find(v.begin(), v.end(), nid) == v.end()) v.push_back(nid);
        };

        if (d.px > 1) {
            add_unique((bx - 1 + d.px) % d.px, by, bz);
            add_unique((bx + 1) % d.px, by, bz);
        }
        if (d.py > 1) {
            add_unique(bx, (by - 1 + d.py) % d.py, bz);
            add_unique(bx, (by + 1) % d.py, bz);
        }
        if (d.pz > 1) {
            add_unique(bx, by, (bz - 1 + d.pz) % d.pz);
            add_unique(bx, by, (bz + 1) % d.pz);
        }
    }

    return d;
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#endif
}

inline void update_block(const std::vector<double>& src,
                         std::vector<double>& dst,
                         const Block& b,
                         const Params& p,
                         double dt) {
    const int N = p.N;
    const int tile = p.tile;

    const double dx = 1.0 / static_cast<double>(N);
    const double dy = dx;
    const double dz = dx;

    const double cx = p.a * dt / dx;
    const double cy = p.b * dt / dy;
    const double cz = p.c * dt / dz;

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

                            const std::size_t center = idx3(i, j, k, N);
                            const double u = src[center];
                            double out = u;

                            if (p.a >= 0.0) {
                                out -= cx * (u - src[idx3(im, j, k, N)]);
                            } else {
                                out -= cx * (src[idx3(ip, j, k, N)] - u);
                            }

                            if (p.b >= 0.0) {
                                out -= cy * (u - src[idx3(i, jm, k, N)]);
                            } else {
                                out -= cy * (src[idx3(i, jp, k, N)] - u);
                            }

                            if (p.c >= 0.0) {
                                out -= cz * (u - src[idx3(i, j, km, N)]);
                            } else {
                                out -= cz * (src[idx3(i, j, kp, N)] - u);
                            }

                            dst[center] = out;
                        }
                    }
                }
            }
        }
    }
}

struct ErrorMetrics {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
};

inline ErrorMetrics compute_errors(const std::vector<double>& U, const Params& p, double final_time) {
    const int N = p.N;
    const double xc = wrap01(p.x0 + p.a * final_time);
    const double yc = wrap01(p.y0 + p.b * final_time);
    const double zc = wrap01(p.z0 + p.c * final_time);

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
                const double exact = periodic_gaussian(x, y, z, xc, yc, zc, p.D);
                const double e = std::fabs(U[idx3(i, j, k, N)] - exact);
                l1 += e;
                l2 += e * e;
                linf = std::max(linf, e);
            }
        }
    }

    const double total = static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    ErrorMetrics err;
    err.l1 = l1 / total;
    err.l2 = std::sqrt(l2 / total);
    err.linf = linf;
    return err;
}

inline void print_report(const char* variant,
                         const Params& p,
                         int workers,
                         int px,
                         int py,
                         int pz,
                         double dt,
                         double elapsed,
                         const ErrorMetrics& err) {
    std::cout << std::setprecision(16);
    std::cout << "Variant : " << variant << "\n";
    std::cout << "N : " << p.N << "\n";
    std::cout << "T : " << p.steps << "\n";
    std::cout << "TILE : " << p.tile << "\n";
    std::cout << "Workers : " << workers << "\n";
    std::cout << "Decomp : " << px << " x " << py << " x " << pz << "\n";
    std::cout << "a : " << p.a << "\n";
    std::cout << "b : " << p.b << "\n";
    std::cout << "c : " << p.c << "\n";
    std::cout << "CFL : " << p.cfl << "\n";
    std::cout << "dt : " << dt << "\n";
    std::cout << "final_time : " << dt * static_cast<double>(p.steps) << "\n";
    std::cout << "L1_error : " << err.l1 << "\n";
    std::cout << "L2_error : " << err.l2 << "\n";
    std::cout << "Linf_error : " << err.linf << "\n";
    std::cout << "Tempo : " << elapsed << " s\n";
}

inline int openmp_naive_main(const char* variant_name) {
    const Params p = read_params();
    const std::size_t total = static_cast<std::size_t>(p.N) * p.N * p.N;
    std::vector<double> U0(total);
    std::vector<double> U1(total);

    initialize(U0, p);
    const double dt = compute_dt(p);

    const int P = omp_get_max_threads();
    const Decomp decomp = build_decomp(P, p.N);

    const double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(P)
    {
        const int tid = omp_get_thread_num();
        const Block block = decomp.blocks[tid];

        for (int step = 0; step < p.steps; ++step) {
            const std::vector<double>& src = (step % 2 == 0) ? U0 : U1;
            std::vector<double>& dst = (step % 2 == 0) ? U1 : U0;

            update_block(src, dst, block, p, dt);

            #pragma omp barrier
        }
    }

    const double elapsed = omp_get_wtime() - t0;

    const std::vector<double>& finalU = (p.steps % 2 == 0) ? U0 : U1;
    const ErrorMetrics err = compute_errors(finalU, p, dt * static_cast<double>(p.steps));
    print_report(variant_name, p, P, decomp.px, decomp.py, decomp.pz, dt, elapsed, err);
    return 0;
}

struct alignas(64) PaddedAtomicInt {
    std::atomic<int> value;
    PaddedAtomicInt() : value(0) {}
};

inline int load_progress(const std::atomic<int>& a) {
    return a.load(std::memory_order_acquire);
}

inline void store_progress(std::atomic<int>& a, int v) {
    a.store(v, std::memory_order_release);
}

inline int load_progress(const PaddedAtomicInt& a) {
    return a.value.load(std::memory_order_acquire);
}

inline void store_progress(PaddedAtomicInt& a, int v) {
    a.value.store(v, std::memory_order_release);
}

template <typename ProgressVector>
inline void wait_neighbors_progress(ProgressVector& progress,
                                    const std::vector<int>& neighbors,
                                    int required_step_count) {
    for (const int nb : neighbors) {
        while (load_progress(progress[nb]) < required_step_count) {
            cpu_relax();
        }
    }
}

template <typename ProgressVector>
inline int openmp_busywait_impl(const char* variant_name) {
    const Params p = read_params();
    const std::size_t total = static_cast<std::size_t>(p.N) * p.N * p.N;
    std::vector<double> U0(total);
    std::vector<double> U1(total);

    initialize(U0, p);
    const double dt = compute_dt(p);

    const int P = omp_get_max_threads();
    const Decomp decomp = build_decomp(P, p.N);

    ProgressVector progress(P);
    for (int i = 0; i < P; ++i) store_progress(progress[i], 0);

    const double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(P)
    {
        const int tid = omp_get_thread_num();
        const Block block = decomp.blocks[tid];
        const auto& neighbors = decomp.neighbors[tid];

        for (int step = 0; step < p.steps; ++step) {
            wait_neighbors_progress(progress, neighbors, step);

            const std::vector<double>& src = (step % 2 == 0) ? U0 : U1;
            std::vector<double>& dst = (step % 2 == 0) ? U1 : U0;

            update_block(src, dst, block, p, dt);
            store_progress(progress[tid], step + 1);
        }
    }

    const double elapsed = omp_get_wtime() - t0;

    const std::vector<double>& finalU = (p.steps % 2 == 0) ? U0 : U1;
    const ErrorMetrics err = compute_errors(finalU, p, dt * static_cast<double>(p.steps));
    print_report(variant_name, p, P, decomp.px, decomp.py, decomp.pz, dt, elapsed, err);
    return 0;
}

inline int openmp_busywait_main(const char* variant_name) {
    return openmp_busywait_impl<std::vector<std::atomic<int>>>(variant_name);
}

inline int openmp_busywait_nofs_main(const char* variant_name) {
    return openmp_busywait_impl<std::vector<PaddedAtomicInt>>(variant_name);
}

inline void checked_sem_wait(sem_t* s) {
    while (sem_wait(s) == -1) {
        if (errno == EINTR) continue;
        perror("sem_wait");
        std::exit(1);
    }
}

inline void checked_sem_post(sem_t* s) {
    if (sem_post(s) != 0) {
        perror("sem_post");
        std::exit(1);
    }
}

struct alignas(64) PaddedSem {
    sem_t sem;
};

inline sem_t* sem_ptr(std::vector<sem_t>& sems, int P, int receiver, int sender) {
    return &sems[static_cast<std::size_t>(receiver) * P + sender];
}

inline sem_t* sem_ptr(std::vector<PaddedSem>& sems, int P, int receiver, int sender) {
    return &sems[static_cast<std::size_t>(receiver) * P + sender].sem;
}

template <typename SemVector>
inline int openmp_sem_impl(const char* variant_name) {
    const Params p = read_params();
    const std::size_t total = static_cast<std::size_t>(p.N) * p.N * p.N;
    std::vector<double> U0(total);
    std::vector<double> U1(total);

    initialize(U0, p);
    const double dt = compute_dt(p);

    const int P = omp_get_max_threads();
    const Decomp decomp = build_decomp(P, p.N);

    SemVector sems(static_cast<std::size_t>(P) * P);
    for (std::size_t i = 0; i < sems.size(); ++i) {
        sem_t* s = nullptr;
        if constexpr (std::is_same<SemVector, std::vector<sem_t>>::value) {
            s = &sems[i];
        } else {
            s = &sems[i].sem;
        }

        if (sem_init(s, 0, 0) != 0) {
            perror("sem_init");
            std::exit(1);
        }
    }

    const double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(P)
    {
        const int tid = omp_get_thread_num();
        const Block block = decomp.blocks[tid];
        const auto& neighbors = decomp.neighbors[tid];

        for (int step = 0; step < p.steps; ++step) {
            if (step > 0) {
                for (const int nb : neighbors) {
                    checked_sem_wait(sem_ptr(sems, P, tid, nb));
                }
            }

            const std::vector<double>& src = (step % 2 == 0) ? U0 : U1;
            std::vector<double>& dst = (step % 2 == 0) ? U1 : U0;

            update_block(src, dst, block, p, dt);

            for (const int nb : neighbors) {
                checked_sem_post(sem_ptr(sems, P, nb, tid));
            }
        }
    }

    const double elapsed = omp_get_wtime() - t0;

    for (std::size_t i = 0; i < sems.size(); ++i) {
        sem_t* s = nullptr;
        if constexpr (std::is_same<SemVector, std::vector<sem_t>>::value) {
            s = &sems[i];
        } else {
            s = &sems[i].sem;
        }
        sem_destroy(s);
    }

    const std::vector<double>& finalU = (p.steps % 2 == 0) ? U0 : U1;
    const ErrorMetrics err = compute_errors(finalU, p, dt * static_cast<double>(p.steps));
    print_report(variant_name, p, P, decomp.px, decomp.py, decomp.pz, dt, elapsed, err);
    return 0;
}

inline int openmp_sem_main(const char* variant_name) {
    return openmp_sem_impl<std::vector<sem_t>>(variant_name);
}

inline int openmp_sem_nofs_main(const char* variant_name) {
    return openmp_sem_impl<std::vector<PaddedSem>>(variant_name);
}

} // namespace fv3d

#endif // FV3D_ADV_COMMON_HPP
