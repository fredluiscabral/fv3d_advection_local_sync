#include "fv3d_adv_diag_common.hpp"

int main() {
    const fv3d::Params p = fv3d::read_params();
    const double dt = fv3d::compute_dt(p);
    const int P = omp_get_max_threads();
    const fv3d::Decomp d = fv3d::build_decomp(P, p.N);

    std::vector<fv3d::PaddedAtomicInt> progress(P);
    for (int i = 0; i < P; ++i) fv3d::store_progress(progress[i], 0);

    volatile int sink = 0;
    const double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(P) shared(sink)
    {
        const int tid = omp_get_thread_num();
        const auto& neighbors = d.neighbors[tid];
        int local = tid;

        for (int step = 0; step < p.steps; ++step) {
            fv3d::wait_neighbors_progress(progress, neighbors, step);
            local += static_cast<int>(neighbors.size()) + (step & 1);
            fv3d::store_progress(progress[tid], step + 1);
        }

        #pragma omp atomic
        sink += local;
    }

    const double elapsed = omp_get_wtime() - t0;
    fv3d_diag::print_zero_error_report("sync3d_busywait_6neighbors_only", p, P, d.px, d.py, d.pz, dt, elapsed);
    return sink == -1 ? 1 : 0;
}
