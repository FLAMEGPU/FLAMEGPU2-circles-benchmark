# Sample Data and figures

## Data
The sample data in this folder was generated using:

+ CUDA 11.2
+ NVIDIA Titan V GPU
+ 100 iterations
+ FLAME GPU commit `@todo`
+ Repository commit `@todo`


## Figures 

Generated using `plot.py`. From this directory, execute:

```
python3 ../plot.py data/ -o figures/ -f 
```

Which output the following to `stdout`.
The RTC cache had not been cleared prior to execution.
```
fixed-density_perSimulationCSV.csv: max_ms_rtc 8.493, mean_ms_rtc 3.990
variable-density_perSimulationCSV.csv: max_ms_rtc 8.966, mean_ms_rtc 3.989

```

The generated figures:

+ `figures/fixed-density--volume--step-ms--model--all.png`
    + Fixed density value of `1.0f` - the agent count is the volume.
    + Shows all 4 implementations
        + Bruteforce is much slower than spatial at large scales - it reads a lot more messages.
        + RTC is faster than non-RTC, due to curve implementation
            + This is compounded by the number of messages read
+ `figures/fixed-density--volume--step-ms--model--zoomed.png`
    + The same as the previous figure, just zoomed in to clearly show the spatial benchmarking.
+ `figures/variable-density--densit--step-ms--volume--3drtc.png`
    + `circles_spatial3d_rtc` only 
    + Varied density - shown on the X axis
    + Varied volume - shown by hue and marker
    + agent count is volume * density.
+ `figures/variable-density--volume--step-ms--density--3drtc.png`
    + `circles_spatial3d_rtc` only 
    + Varied density - shown by hue and marker
    + Varied volume - shown on the X axis
    + agent count is volume * density.


Communication / interaction radius is fixed to `2.0f` for all benchmarks, corresponding to a spherical volume of `~33.5`, and a moores neighbourhood of `216`?


## Key points 

+ Repetition of 3 runs, different seeds.
+ Initialised in 3D space using uniform distribution (rng).
+ Over time, local density changes agents move, leading to higher density areas with lower performance. 
+ Titan V would be ~ 160k threads peak, but 50% occupancy for some kernels, so changes in gradient at ~80k.
+ Larger populations above this require multiple waves to complete, but spatial is non-linear
    + Larger environments have larger bin counts, PBM construction takes more time
    + local density can increase beyond initial values.
        + Average message count over all steps is ~ 200 (see csv's)
+ 0th iteration includes some one-time costs, otherwise its relatively stable over 100 iterations
    + Once the model converges runtime changes based on how dense the densest areas are.