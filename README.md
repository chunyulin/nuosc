A typical workflow on ARM/TWCC cluster

1. Use ". ./gocom" to make.
1. Submit multiple runs with "run*.sh" into slurm. Each set of runs is in the `runtag` folder.
1. Post processing:
    1. For each 1D/2D run, use `../plt_analysis.py` to generate comparison plot for different runs of statistics quantities `|P|-1, Pee, M0` in `./*/analysis.dat`.
    1. For each 1D/2D run, use `../cpltZVz_P3.py` to generate comparison plot of P(Z,Vz) for different run. Assuming each run has the same DUMP period.
    1. For each 2D run, use `../../pltYZ_P3.py *.bin` to generate contour plot.
    1. Generate movie by `makemovie.*`.
--
