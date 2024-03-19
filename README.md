# NuOSC

## Baseline test on T4:

1. Compile ". ./gocom"
2. See bench/ for batch submission, result collection, and compare plot.
3. Weak scaling upto 512 node x 112 core tested via OmeAPI 2024, with almost perfect weak scaling for 8-rank per node.

|   Normalized time per step-grid (ns)   |   8-rank  |
| --- | --- | --- |
|   FD, 2-flavor (8 vars)   |   10   |
|   FD, 3-flavor (18 vars)   |   22   |
|   FV, 2-flavor (8 vars)   |   25   |
|   FV, 3-flavor (18 vars)   |   55-75   |


TODO:
- Check results for SYNC_NCCL and SYNC_MPI_ONESIDE_COPY, which are incorrects on A100 test node.

## Simple benchmark on Neoverse-N1 with A100
- 3D: via `make test3d`. Problem size comparable to host/device memory. Highest opt complied with IEEE974 std.

|   Time per step-grid (ns)   |   80-core (Neo N1)   |   A100 x 1   |
| --- | --- | --- |
|   FD, 2-flavor (8 vars)   |   37   |   4  (9x)   |
|   FD, 3-flavor (18 vars)   |   85   |   9  (9x)   |
|   FV, 2-flavor (8 vars)   |   100   |   9  (11x)   |
|   FV, 3-flavor (18 vars)   |   235   |   21  (11x)   |

- Eariler 1D test has difference tendency, maybe due to implementation and lower arithmetic intgensity.

|   Time per step-grid (ns)   |   80-core (Neo N1)   |   A100 x 1   |
| --- | --- | --- |
|   FD, 2-flavor |   16   |    3    (5x)   |
|   FV, 2-flavor |   20   |   10    (2x)   |

## Typical eariler workflow on ARM/TWCC cluster

1) Use ". ./gocom" to make.
2) Sinale run by ./nuosc <argument>
3) Multiple runs by "run*.sh" submitting to Slurm or nohup. Each set of runs is in the `runtag` folder.
4) Make plots:
    a) In each runtag folder, use `plt_analysis.py` to compare statistics quantities like `|P|-1, Pee, M0` for different runs.
    b) In each run, use `plt_P3.py *.bin` to generate P3 plots.
    c) The animation can be generated via `makemovie`.


For preliminary mpi run w/o full output function, 
try "make test2d_mpi" with additional argument [./nuosc --np px py] to specify decomposition geometry.
