# FV3D diagnostics package

Este pacote adiciona as versões que faltavam para investigar por que o MPI 3D ficou melhor que as variantes OpenMP globais.

## Novas variantes numéricas

- `fv3d_adv_omp_mpilike.cpp`
  - OpenMP com um subdomínio 3D local por thread.
  - Cada thread tem `U0/U1` próprios com halos nas 6 faces.
  - A troca de halos é feita por cópia direta em memória compartilhada.
  - Sincronização por `PaddedAtomicInt`, como na busy-wait tratada.
  - Testa se a vantagem do MPI vem de layout local + halos.

- `fv3d_adv_omp_naive_firsttouch_same_decomp.cpp`
  - Ainda usa arrays globais, mas aloca sem inicialização e faz first-touch pela mesma decomposição 3D usada no kernel.
  - Mantém barreira global por passo.
  - Testa NUMA/first-touch contra a naive original.

- `fv3d_adv_omp_busywait_firsttouch_same_decomp.cpp`
  - Arrays globais, first-touch pela mesma decomposição 3D e sincronização local busy-wait padded.
  - Testa se a diferença para MPI vem mais de first-touch/layout global ou de halos locais.

## Microbenchmarks de sincronização

- `sync3d_barrier_only.cpp`
  - Mede custo de uma barreira global por passo.

- `sync3d_busywait_6neighbors_only.cpp`
  - Mede custo de sincronização local por progresso usando a mesma vizinhança 3D.

- `sync3d_sem_6neighbors_only.cpp`
  - Mede custo de sincronização local por semáforos usando a mesma vizinhança 3D.

## Compilação

```bash
module load amd-compilers/5.0.0
module load amd-libraries/5.0.0
module load openmpi/amd/5.0
make all
```

Ou somente diagnósticos:

```bash
make diag
```

## Testes sugeridos

Comece com:

```text
N = 256
T = 100
TILE = 32
```

Compare em 64, 128 e 192 workers:

```text
mpi_puro
omp_busywait_nofs
omp_mpilike_3d
omp_naive_firsttouch_same_decomp
omp_busywait_firsttouch_same_decomp
```

Depois rode os microbenchmarks:

```text
sync3d_barrier_only
sync3d_busywait_6neighbors_only
sync3d_sem_6neighbors_only
```

A interpretação principal é:

- se `omp_mpilike_3d` aproximar do MPI, o fator dominante é layout local/halos;
- se `firsttouch_same_decomp` aproximar do MPI, o fator dominante é NUMA/first-touch;
- se os microbenchmarks forem caros, a sincronização 3D com até 6 vizinhos está pesando;
- se os microbenchmarks forem baratos, o gargalo está no kernel/memória/layout.
