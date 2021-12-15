# Sample Data and figures

## Data
The sample data for fixed and variable densities in this folder was generated using:

+ Tesla V100-SXM2-32GB
+ NVIDIA Driver 460.32.03
+ CUDA 10.2.98
+ GCC 8.3.0
+ CentOS 7.9.2009
+ FLAME GPU 2 @ [`d238333`](https://github.com/FLAMEGPU/FLAMEGPU2/tree/d238333)

The sample data for variable communication radius in this folder was generated using:

+ Titan V 12GB
+ NVIDIA Driver 455.23.05
+ CUDA 10.2.89
+ GCC 7.5.0
+ Ubuntu 16.04.7 LTS

## Figures 

Generated using `plot.py`. From this directory, execute:

```
python3 ../plot.py data/ -o figures/ -f 
```

Which output the following to `stdout`.
```
fixed-density_perSimulationCSV.csv: max_s_rtc 12919.013, mean_s_rtc 118.152
variable-density_perSimulationCSV.csv: max_s_rtc 18.127, mean_s_rtc 6.395
comm-radius_perSimulationCSV.csv: max_s_rtc 9264.425, mean_s_rtc 210.166

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
+ `figures/comm-radius--lineplot-spatial3D-bruteforce-rtc-only.png`
    + `circles_spatial3d_rtc` and `circles_bruteforce_rtc`
    + Varied communication radius - shown on the X axis
    + Model differentiated by marker
    + Agent count is fixed at 64,000
    + Environment width is fixed at 40
    + Agent Density is `1.0f`


Communication / interaction radius is fixed to `2.0f` for fixed density and variable density benchmarks, corresponding to a spherical volume of `~33.5`, and a moores neighbourhood of `216`?


## Key points 

+ Repetition of 3 runs, different seeds.
+ Initialised in 3D space using uniform distribution (rng).
+ Over time, local density changes agents move, leading to higher density areas with lower performance. 
+ Titan V would be ~ 160k threads peak, but 50% occupancy for some kernels, so changes in gradient at ~80k.
+ Larger populations above this require multiple waves to complete, but spatial is non-linear
    + Larger environments have larger bin counts, PBM construction takes more time
    + local density can increase beyond initial values.
+ 0th iteration includes some one-time costs, otherwise its relatively stable over 200 iterations
    + Once the model converges runtime changes based on how dense the densest areas are.