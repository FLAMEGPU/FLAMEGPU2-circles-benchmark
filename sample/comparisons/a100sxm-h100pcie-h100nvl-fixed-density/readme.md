## Commands

```bash
python3 plot.py -i sample/data/a100-550.163.01/2.0.0-rc.4-a100-sxm4-12.4-beltsoff/ -o sample/figures/a100-550.163.01/2.0.0-rc.4-a100-sxm4-12.4-beltsoff/
python3 plot.py -i sample/data/h100pcie-550.163.01/2.0.0-rc.4-a100-pcie-12.4-beltsoff/ -o sample/figures/h100pcie-550.163.01/2.0.0-rc.4-a100-pcie-12.4-beltsoff/
python3 plot.py -i sample/data/h100-nvl-550.163.01/2.0.0-rc.4-a100-nvl-12.4-beltsoff/ sample/figures/h100-nvl-550.163.01/2.0.0-rc.4-a100-nvl-12.4-beltsoff/
```

## output

```bash
python3 plot_comparisons.py -c sample/comparisons/a100sxm-h100pcie-h100nvl-fixed-density/config.yml 
```

`mean_s_simulation for agent_count == 1000000`


| model                  |   A100 SXM4 CUDA 12.4 |   H100 PCIe CUDA 12.4 |   H100 NVL  CUDA 12.4 |
|:-----------------------|----------------------:|----------------------:|----------------------:|
| circles_bruteforce     |              1070.663 |               801.966 |               661.511 |
| circles_bruteforce_rtc |               520.629 |               432.659 |               357.848 |
| circles_spatial3D      |                 0.706 |                 0.527 |                 0.454 |
| circles_spatial3D_rtc  |                 0.560 |                 0.426 |                 0.367 |

mean_s_simulation__speedup__A100_SXM4_CUDA_12.4 for agent_count == 1000000
| model                  |   A100 SXM4 CUDA 12.4 |   H100 PCIe CUDA 12.4 |   H100 NVL  CUDA 12.4 |
|:-----------------------|----------------------:|----------------------:|----------------------:|
| circles_bruteforce     |                 1.000 |                 1.335 |                 1.619 |
| circles_bruteforce_rtc |                 1.000 |                 1.203 |                 1.455 |
| circles_spatial3D      |                 1.000 |                 1.339 |                 1.553 |
| circles_spatial3D_rtc  |                 1.000 |                 1.314 |                 1.525 |
