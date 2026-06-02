#include "fv3d_adv_diag_common.hpp"

int main() {
    const fv3d::Params p = fv3d::read_params();
    const double dt = fv3d::compute_dt(p);
    const int P = omp_get_max_threads();
    const fv3d::Decomp d = fv3d::build_decomp(P, p.N);

    volatile int sink = 0;
    const double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(P) shared(sink)
    {
        int local = omp_get_thread_num();
        for (int step = 0; step < p.steps; ++step) {
            local += step & 1;
            #pragma omp barrier
        }
        #pragma omp atomic
        sink += local;
    }

    const double elapsed = omp_get_wtime() - t0;
    fv3d_diag::print_zero_error_report("sync3d_barrier_only", p, P, d.px, d.py, d.pz, dt, elapsed);
    return sink == -1 ? 1 : 0;
}
