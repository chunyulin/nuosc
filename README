# NuOSC

## Baseline test on T4:

1. Compile ". ./gocom"
2. See bench/ for batch submission, result collection, and compare plot.

Note: 
- Weak scaling upto 512 rank x 112 core ~ 70%. MPI one-sided copy has better scaling but slower for fewer ranks.
- Max size within a node is roughly 140x140x140x18x18 for 2-flavor case.
- FD-code essentailly memory bound that 112-core has little marginal speedup over 56-core in a node. FV version should be more compute intensive and utiliztion.
- On T4, mpich and intel mpi has the same performance. OpenMPI seems has trouble to initize large rank.


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
