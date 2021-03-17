## Benchmark results of 3D TOF/NONTOF Joseph projector

scanner geometry: 28/5 modules with 16/9 crystals per module
full TOF sinogram size: (357,224,2025,27) -> no spanning
subset sino size: (357,28,2025,27)
TOF FWHM: 60mm (400ps)
voxel size: 2x2x2 mm
FOV: 250mm (brain) or 600mm (WB)


### NVIDIA Titan XP on Windows using CUDA 11 - approximate SIGNA geometry

#### Sinogram 1 out of 28 subsets

|  | Non-TOF brain (s) | Non-TOF WB (s) | TOF brain (s) | TOF WB (s) |
| -- | -- | -- | -- | -- |
| fwd   | 0.070 +-  0.007  | 0.113 +-  0.006  | 0.361 +-  0.001 | 0.704 +-  0.001 |
| back | 0.085 +-  0.007  | 0.223 +-  0.006  | 0.468 +-  0.001 | 1.586 +-  0.007 |

#### LM 4e6 events

|  | Non-TOF brain (s) | Non-TOF WB (s) | TOF brain (s) | TOF WB (s) |
| -- | -- | -- | -- | -- |
| fwd   |  0.217 +-  0.007 | 0.873 +-  0.012  | 0.147 +-  0.007  | 0.260 +-  0.006 |
| back |  0.806 +-  0.009 | 3.425 +-  0.004  | 0.388 +-  0.007  | 0.858 +-  0.003 |

#### LM 4e5 events

|  | Non-TOF brain (s) | Non-TOF WB (s) | TOF brain (s) | TOF WB (s) |
| -- | -- | -- | -- | -- |
| fwd   | 0.021 +-  0.007 | 0.087 +-  0.007 | 0.024 +-  0.007 | 0.040 +-  0.007 |
| back | 0.053 +-  0.007 | 0.328 +-  0.001 | 0.046 +-  0.001 | 0.128 +-  0.005 |

---

ngpus:1 counts:4000000.0 nsubsets:28 n:5 tpb:64 nontof:True img_mem_order:C sino_dim_order:['0', '1', '2'] fov:brain voxsize:['2', '2', '2']

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

sino fwd  0.0705 s (mean) +-  0.0078 s (std)
sino back 0.0859 s (mean) +-  0.0076 s (std)

generating LM data

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

LM fwd   0.2171 s (mean) +-  0.0072 s (std)
LM back  0.8064 s (mean) +-  0.0099 s (std)

ngpus:1 counts:4000000.0 nsubsets:28 n:5 tpb:64 nontof:True img_mem_order:C sino_dim_order:['0', '1', '2'] fov:wb voxsize:['2', '2', '2']

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

sino fwd  0.1138 s (mean) +-  0.0064 s (std)
sino back 0.2231 s (mean) +-  0.0065 s (std)

generating LM data

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

LM fwd   0.8738 s (mean) +-  0.0120 s (std)
LM back  3.4250 s (mean) +-  0.0046 s (std)

ngpus:1 counts:4000000.0 nsubsets:28 n:5 tpb:64 nontof:False img_mem_order:C sino_dim_order:['0', '1', '2'] fov:brain voxsize:['2', '2', '2']

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

sino fwd  0.3611 s (mean) +-  0.0013 s (std)
sino back 0.4687 s (mean) +-  0.0003 s (std)

generating LM data

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

LM fwd   0.1470 s (mean) +-  0.0074 s (std)
LM back  0.3887 s (mean) +-  0.0071 s (std)

ngpus:1 counts:4000000.0 nsubsets:28 n:5 tpb:64 nontof:False img_mem_order:C sino_dim_order:['0', '1', '2'] fov:wb voxsize:['2', '2', '2']

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

sino fwd  0.7047 s (mean) +-  0.0016 s (std)
sino back 1.5864 s (mean) +-  0.0072 s (std)

generating LM data

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

LM fwd   0.2605 s (mean) +-  0.0065 s (std)
LM back  0.8582 s (mean) +-  0.0030 s (std)

ngpus:1 counts:400000.0 nsubsets:28 n:5 tpb:64 nontof:True img_mem_order:C sino_dim_order:['0', '1', '2'] fov:brain voxsize:['2', '2', '2']

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

sino fwd  0.0663 s (mean) +-  0.0067 s (std)
sino back 0.0822 s (mean) +-  0.0067 s (std)

generating LM data

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

LM fwd   0.0216 s (mean) +-  0.0076 s (std)
LM back  0.0534 s (mean) +-  0.0076 s (std)

ngpus:1 counts:400000.0 nsubsets:28 n:5 tpb:64 nontof:True img_mem_order:C sino_dim_order:['0', '1', '2'] fov:wb voxsize:['2', '2', '2']

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

sino fwd  0.1134 s (mean) +-  0.0076 s (std)
sino back 0.2194 s (mean) +-  0.0120 s (std)

generating LM data

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

LM fwd   0.0877 s (mean) +-  0.0076 s (std)
LM back  0.3288 s (mean) +-  0.0009 s (std)

ngpus:1 counts:400000.0 nsubsets:28 n:5 tpb:64 nontof:False img_mem_order:C sino_dim_order:['0', '1', '2'] fov:brain voxsize:['2', '2', '2']

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

sino fwd  0.3566 s (mean) +-  0.0062 s (std)
sino back 0.4688 s (mean) +-  0.0112 s (std)

generating LM data

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

LM fwd   0.0249 s (mean) +-  0.0075 s (std)
LM back  0.0469 s (mean) +-  0.0001 s (std)

ngpus:1 counts:400000.0 nsubsets:28 n:5 tpb:64 nontof:False img_mem_order:C sino_dim_order:['0', '1', '2'] fov:wb voxsize:['2', '2', '2']

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

sino fwd  0.7057 s (mean) +-  0.0083 s (std)
sino back 1.5991 s (mean) +-  0.0057 s (std)

generating LM data

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

LM fwd   0.0409 s (mean) +-  0.0076 s (std)
LM back  0.1287 s (mean) +-  0.0059 s (std)

ngpus:1 counts:40000.0 nsubsets:28 n:5 tpb:64 nontof:True img_mem_order:C sino_dim_order:['0', '1', '2'] fov:brain voxsize:['2', '2', '2']

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

sino fwd  0.0704 s (mean) +-  0.0079 s (std)
sino back 0.0830 s (mean) +-  0.0065 s (std)

generating LM data

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

LM fwd   0.0094 s (mean) +-  0.0077 s (std)
LM back  0.0031 s (mean) +-  0.0062 s (std)

ngpus:1 counts:40000.0 nsubsets:28 n:5 tpb:64 nontof:True img_mem_order:C sino_dim_order:['0', '1', '2'] fov:wb voxsize:['2', '2', '2']

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

sino fwd  0.1172 s (mean) +-  0.0078 s (std)
sino back 0.2226 s (mean) +-  0.0070 s (std)

generating LM data

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

LM fwd   0.0217 s (mean) +-  0.0076 s (std)
LM back  0.0538 s (mean) +-  0.0083 s (std)

ngpus:1 counts:40000.0 nsubsets:28 n:5 tpb:64 nontof:False img_mem_order:C sino_dim_order:['0', '1', '2'] fov:brain voxsize:['2', '2', '2']

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

sino fwd  0.3561 s (mean) +-  0.0074 s (std)
sino back 0.4700 s (mean) +-  0.0020 s (std)

generating LM data

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

LM fwd   0.0062 s (mean) +-  0.0077 s (std)
LM back  0.0125 s (mean) +-  0.0063 s (std)

ngpus:1 counts:40000.0 nsubsets:28 n:5 tpb:64 nontof:False img_mem_order:C sino_dim_order:['0', '1', '2'] fov:wb voxsize:['2', '2', '2']

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

sino fwd  0.7083 s (mean) +-  0.0062 s (std)
sino back 1.5920 s (mean) +-  0.0038 s (std)

generating LM data

run 1 / 5
run 2 / 5
run 3 / 5
run 4 / 5
run 5 / 5

LM fwd   0.0151 s (mean) +-  0.0010 s (std)
LM back  0.0442 s (mean) +-  0.0067 s (std)

