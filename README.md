# NuOSC

## Note:
- Add spatial avg monitoring (WriteBinary), to be improved by HDF.

### 240420
- Fix icosahedra integral weight by 2\pi -- reducible to 1d.
- Use std::round instead of type-casting in determining nx[] to avoid unexpected result.
- Fix unit test for sphere integral: test_v2dint.cpp
- Reorganize file structure of unittest and script.
- Switchable between real = double | float.

## Status

- Current 3D status: 2/3 Flavor, FD8/WENO, OpenMP/OpenACC
    - FD8 is 20% faster than WENO, with the same phyical result, but with O(10^2) deviation of |P|.
    - FD4/FD8 also pass code comparisom case.
    - Compressed (GZIP) checkpointing.
- Long 3D box in X/Y/Z direction reduces to 1D box.
    - Density matrix and |M0| of 3D-Z-BOX are FP64 bit-reproducible to 1D.
    - Vz-phi grid is as good/fast as Icosahedra grid for X-box case. (vz-phi grid is perfectly fit for Z-box case).
    - Note about numerical normalization vs analytical normalization for bit-preservation.
- Some facts/observations:
    - Time for A100 ~10ns per step-grid, which is ~30s per simulation time for 800x8x8x643 grid with dt=0.05 (2-flavo
    - FP32 run is 2x faster than FP64 with similar physical outcome with expected O(-7) conservation.
    - A naive implementation of complex number treatment on the flavor field is slower.
    - Low-storage RK3 scheme supported (need 3x copy instead of 4x).
- TODO:
    - HDF5 checkpointing.
    - GPU-offloading of size large than device memory.


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
