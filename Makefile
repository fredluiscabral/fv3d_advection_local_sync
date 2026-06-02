CXX      ?= g++
MPICXX   ?= mpicxx

CXXFLAGS ?= -O3 -std=c++17 -march=native -Wall -Wextra
OMPFLAGS ?= -fopenmp
LDFLAGS  ?= -pthread

COMMON = fv3d_adv_common.hpp
DIAG_COMMON = fv3d_adv_diag_common.hpp

OMP_TARGETS = \
	fv3d_adv_omp_naive \
	fv3d_adv_omp_naive_nofs \
	fv3d_adv_omp_busywait_nobarrier \
	fv3d_adv_omp_busywait_nobarrier_nofs \
	fv3d_adv_omp_sem_nobarrier \
	fv3d_adv_omp_sem_nobarrier_nofs

MPI_TARGETS = fv3d_adv_mpi_naive_3d

DIAG_TARGETS = \
	fv3d_adv_omp_mpilike \
	fv3d_adv_omp_naive_firsttouch_same_decomp \
	fv3d_adv_omp_busywait_firsttouch_same_decomp \
	sync3d_barrier_only \
	sync3d_busywait_6neighbors_only \
	sync3d_sem_6neighbors_only

.PHONY: all openmp mpi diag clean

all: openmp mpi diag

openmp: $(OMP_TARGETS)

mpi: $(MPI_TARGETS)

diag: $(DIAG_TARGETS)

fv3d_adv_omp_naive: fv3d_adv_omp_naive.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

fv3d_adv_omp_naive_nofs: fv3d_adv_omp_naive_nofs.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

fv3d_adv_omp_busywait_nobarrier: fv3d_adv_omp_busywait_nobarrier.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

fv3d_adv_omp_busywait_nobarrier_nofs: fv3d_adv_omp_busywait_nobarrier_nofs.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

fv3d_adv_omp_sem_nobarrier: fv3d_adv_omp_sem_nobarrier.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

fv3d_adv_omp_sem_nobarrier_nofs: fv3d_adv_omp_sem_nobarrier_nofs.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

fv3d_adv_mpi_naive_3d: fv3d_adv_mpi_naive_3d.cpp $(COMMON)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

fv3d_adv_omp_mpilike: fv3d_adv_omp_mpilike.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

fv3d_adv_omp_naive_firsttouch_same_decomp: fv3d_adv_omp_naive_firsttouch_same_decomp.cpp $(COMMON) $(DIAG_COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

fv3d_adv_omp_busywait_firsttouch_same_decomp: fv3d_adv_omp_busywait_firsttouch_same_decomp.cpp $(COMMON) $(DIAG_COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

sync3d_barrier_only: sync3d_barrier_only.cpp $(COMMON) $(DIAG_COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

sync3d_busywait_6neighbors_only: sync3d_busywait_6neighbors_only.cpp $(COMMON) $(DIAG_COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

sync3d_sem_6neighbors_only: sync3d_sem_6neighbors_only.cpp $(COMMON) $(DIAG_COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(OMP_TARGETS) $(MPI_TARGETS) $(DIAG_TARGETS) *.o
