# fv3d_advection_local_sync

Miniapp em C++ para a equação de advecção linear 3D por volumes finitos explícitos:

\[
u_t + a u_x + b u_y + c u_z = 0.
\]

O domínio é periódico em \([0,1)^3\). A condição inicial é um pulso gaussiano periódico, e a solução exata é a translação do pulso:

\[
u(x,y,z,t)=u_0(x-at,y-bt,z-ct).
\]

Cada executável imprime:

```text
L1_error
L2_error
Linf_error
Tempo : ... s
```

A linha `Tempo :` foi mantida para compatibilidade com os scripts de consolidação usados nos experimentos anteriores.

## Variantes incluídas

- `fv3d_adv_omp_naive`
- `fv3d_adv_omp_naive_nofs`
- `fv3d_adv_omp_busywait_nobarrier`
- `fv3d_adv_omp_busywait_nobarrier_nofs`
- `fv3d_adv_omp_sem_nobarrier`
- `fv3d_adv_omp_sem_nobarrier_nofs`
- `fv3d_adv_mpi_naive_3d`

## Observação sobre `_nofs`

Nas variantes com sincronização local, `_nofs` usa estruturas de coordenação alinhadas/padded para reduzir false sharing. A variante `omp_naive_nofs` é mantida por simetria de nomes com os experimentos anteriores; como a versão naive usa barreira global e não usa flags/semafóros de progresso, ela compartilha o mesmo kernel de blocos da `omp_naive`.

## Decomposição 3D

As variantes OpenMP dividem o domínio em uma grade cartesiana 3D de blocos. Cada thread recebe um bloco. As variantes locais usam dependências entre vizinhos de face do bloco.

A variante MPI usa `MPI_Dims_create` e `MPI_Cart_create` para montar uma decomposição cartesiana 3D periódica, com troca de halos nas 6 faces.

## Compilação

```bash
make openmp
make mpi
make all
```

No SDumont, carregue os módulos usuais antes de compilar:

```bash
module load amd-compilers/5.0.0
module load amd-libraries/5.0.0
module load openmpi/amd/5.0
make all
```

## Execução rápida

OpenMP:

```bash
export OMP_NUM_THREADS=4
./fv3d_adv_omp_busywait_nobarrier_nofs
```

MPI:

```bash
mpirun -np 8 ./fv3d_adv_mpi_naive_3d
```

## Tamanho de malha

Para testes rápidos, use `N=64` ou `N=128`.

Para experimento de desempenho em um nó, `N=512` já representa:

\[
512^3 = 134,217,728
\]

células. Dois campos `double` usam cerca de 2 GiB, sem contar overhead e halos do MPI.

Não use `N=8192` em 3D; isso é inviável em memória.
