## Benchmark results of 3D TOF/NONTOF Joseph projector

- scanner geometry: 28/5 modules with 16/9 crystals per module
- full TOF sinogram size: (357,224,2025,27) -> no spanning
- subset sino size: (357,8,2025,27)
- TOF FWHM: 60mm (400ps)
- voxel size: 2x2x2 mm
- FOV: 250mm (brain) or 600mm (WB)


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
