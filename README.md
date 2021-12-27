A typical workflow on ARM/TWCC cluster

1) Use ". ./gocom" to make.
2) Submit multiple runs with "run*.sh" into slurm. Each set of runs is in the `runtag` folder.
3) Make plots:
    a) In each runtag folder, use `plt_analysis.py` to compare statistics quantities like `|P|-1, Pee, M0` for different runs.
    b) For 1D run, use `plt_skimshot*.py *.bin` to generate plots for each binary snapshot.
    c) For 2D run, use `pltYZ_P3.py *.bin` to generate contour plot.
    d) Generate movie by `makemovie.*`.
--
