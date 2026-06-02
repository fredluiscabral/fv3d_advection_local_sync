#include "fv3d_adv_diag_common.hpp"

int main() {
    const fv3d::Params p = fv3d::read_params();
    const double dt = fv3d::compute_dt(p);
    const int P = omp_get_max_threads();
    const fv3d::Decomp d = fv3d::build_decomp(P, p.N);

    std::vector<fv3d::PaddedSem> sems(static_cast<std::size_t>(P) * P);
    for (std::size_t i = 0; i < sems.size(); ++i) {
        if (sem_init(&sems[i].sem, 0, 0) != 0) {
            perror("sem_init");
            return 1;
        }
    }

    volatile int sink = 0;
    const double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(P) shared(sink)
    {
        const int tid = omp_get_thread_num();
        const auto& neighbors = d.neighbors[tid];
        int local = tid;

        for (int step = 0; step < p.steps; ++step) {
            if (step > 0) {
                for (const int nb : neighbors) {
                    fv3d::checked_sem_wait(fv3d::sem_ptr(sems, P, tid, nb));
                }
            }

            local += static_cast<int>(neighbors.size()) + (step & 1);

            for (const int nb : neighbors) {
                fv3d::checked_sem_post(fv3d::sem_ptr(sems, P, nb, tid));
            }
        }

        #pragma omp atomic
        sink += local;
    }

    const double elapsed = omp_get_wtime() - t0;

    for (std::size_t i = 0; i < sems.size(); ++i) {
        sem_destroy(&sems[i].sem);
    }

    fv3d_diag::print_zero_error_report("sync3d_sem_6neighbors_only", p, P, d.px, d.py, d.pz, dt, elapsed);
    return sink == -1 ? 1 : 0;
}
