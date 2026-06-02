#include "fv3d_adv_diag_common.hpp"

int main() {
    const fv3d::Params p = fv3d::read_params();
    const std::size_t total = static_cast<std::size_t>(p.N) * p.N * p.N;

    fv3d_diag::AlignedDoubles U0;
    fv3d_diag::AlignedDoubles U1;
    if (!U0.allocate(total) || !U1.allocate(total)) {
        std::cerr << "ERRO: falha ao alocar arrays globais alinhados.\n";
        return 1;
    }

    const double dt = fv3d::compute_dt(p);
    const int P = omp_get_max_threads();
    const fv3d::Decomp decomp = fv3d::build_decomp(P, p.N);

    #pragma omp parallel num_threads(P)
    {
        const int tid = omp_get_thread_num();
        fv3d_diag::initialize_global_block(U0.data(), U1.data(), decomp.blocks[tid], p);
    }

    std::vector<fv3d::PaddedAtomicInt> progress(P);
    for (int i = 0; i < P; ++i) fv3d::store_progress(progress[i], 0);

    const double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(P)
    {
        const int tid = omp_get_thread_num();
        const fv3d::Block block = decomp.blocks[tid];
        const auto& neighbors = decomp.neighbors[tid];

        for (int step = 0; step < p.steps; ++step) {
            fv3d::wait_neighbors_progress(progress, neighbors, step);

            const double* src = (step % 2 == 0) ? U0.data() : U1.data();
            double* dst       = (step % 2 == 0) ? U1.data() : U0.data();

            fv3d_diag::update_global_block_raw(src, dst, block, p, dt);
            fv3d::store_progress(progress[tid], step + 1);
        }
    }

    const double elapsed = omp_get_wtime() - t0;
    const double* finalU = (p.steps % 2 == 0) ? U0.data() : U1.data();
    const fv3d::ErrorMetrics err = fv3d_diag::compute_errors_raw(finalU, p, dt * static_cast<double>(p.steps));
    fv3d::print_report("omp_busywait_firsttouch_same_decomp", p, P, decomp.px, decomp.py, decomp.pz, dt, elapsed, err);
    return 0;
}
