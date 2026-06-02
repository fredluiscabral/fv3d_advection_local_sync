# FV3D Advection Local Sync — pacote completo de códigos

Este pacote consolida todas as versões 3D disponíveis neste momento: variantes principais, variantes diagnósticas e microbenchmarks de sincronização.

## Arquivos-fonte incluídos

- `fv3d_adv_common.hpp`
- `fv3d_adv_diag_common.hpp`
- `fv3d_adv_mpi_naive_3d.cpp`
- `fv3d_adv_omp_busywait_firsttouch_same_decomp.cpp`
- `fv3d_adv_omp_busywait_nobarrier.cpp`
- `fv3d_adv_omp_busywait_nobarrier_nofs.cpp`
- `fv3d_adv_omp_mpilike.cpp`
- `fv3d_adv_omp_naive.cpp`
- `fv3d_adv_omp_naive_firsttouch_same_decomp.cpp`
- `fv3d_adv_omp_naive_nofs.cpp`
- `fv3d_adv_omp_sem_nobarrier.cpp`
- `fv3d_adv_omp_sem_nobarrier_nofs.cpp`
- `sync3d_barrier_only.cpp`
- `sync3d_busywait_6neighbors_only.cpp`
- `sync3d_sem_6neighbors_only.cpp`

## Variantes principais

- `fv3d_adv_omp_naive.cpp`
- `fv3d_adv_omp_naive_nofs.cpp`
- `fv3d_adv_omp_busywait_nobarrier.cpp`
- `fv3d_adv_omp_busywait_nobarrier_nofs.cpp`
- `fv3d_adv_omp_sem_nobarrier.cpp`
- `fv3d_adv_omp_sem_nobarrier_nofs.cpp`
- `fv3d_adv_mpi_naive_3d.cpp`

## Variantes diagnósticas

- `fv3d_adv_omp_mpilike.cpp`
- `fv3d_adv_omp_naive_firsttouch_same_decomp.cpp`
- `fv3d_adv_omp_busywait_firsttouch_same_decomp.cpp`

## Microbenchmarks de sincronização

- `sync3d_barrier_only.cpp`
- `sync3d_busywait_6neighbors_only.cpp`
- `sync3d_sem_6neighbors_only.cpp`

## Cabeçalhos

- `fv3d_adv_common.hpp`
- `fv3d_adv_diag_common.hpp`

## Uso típico

```bash
module load amd-compilers/5.0.0
module load amd-libraries/5.0.0
module load openmpi/amd/5.0
make clean
make all
```

Para compilar apenas os diagnósticos, se o Makefile disponível tiver esse alvo:

```bash
make diag
```
