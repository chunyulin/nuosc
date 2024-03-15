# NuOSC

## Baseline test on T4:

1. Compile ". ./gocom"
2. See bench/ for batch submission, result collection, and compare plot.

Note: 
- Weak scaling upto 512 rank x 112 core ~ 70%. MPI one-sided copy has better scaling but slower for fewer ranks.
- Max size within a node is roughly 140x140x140x18x18 for 2-flavor case.
- FD-code essentailly memory bound that 112-core has little marginal speedup over 56-core in a node. FV version should be more compute intensive and utiliztion.
- On T4, mpich and intel mpi has the same performance. OpenMPI seems has trouble to initize large rank.

TODO:
- Check results for SYNC_NCCL and SYNC_MPI_ONESIDE_COPY, which are incorrects on A100 test node.

## Simple benchmark on Neoverse-N1 with A100

- 3D: via `make test3d`

|   Time per step-grid (ns)   |   80-core (Neo N1)   |   A100 x 1   |
| --- | --- | --- |
|   FD, 2-flavor (8 vars)   |   40   |   3   (13x)   |
|   FD, 3-flavor (18 vars)   |   100   |   8  (13x)   |
|   FV, 2-flavor (8 vars)   |   400   |   8  (50x)   |
|   FV, 3-flavor (18 vars)   |   1000   |   20  (50x)   |

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
