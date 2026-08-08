# Delft3D model setup

This folder contains the Delft3D-FLOW model input files used in the manuscript:

> **Size- and Geometry-Dependent Channel–Flat Partitioning in Macrotidal Muddy Basins**

## Model cases

```text
long_basin/   Idealized elongated basin
wide_basin/   Idealized wide basin
```

The two cases have the same idealized basin area and external forcing but different planform geometries. They are used to investigate the effects of basin shape on channel–flat partitioning and basin infilling.

## Files in each case

| File | Description |
| --- | --- |
| `mbf.mdf` | Main Delft3D-FLOW model definition |
| `mbf.mdw` | Delft3D-WAVE/SWAN model definition and wave settings |
| `mbf.bnd` | Open-boundary definitions |
| `mbf.bca` | Tidal boundary amplitudes and phases |
| `mbf.bcc` | Suspended-sediment boundary concentrations |
| `mbf.wnd` | Wind forcing |
| `mbf.sed` | Sediment properties |
| `mbf.mor` | Morphological settings |
| `mbf.obs` | Observation-point definitions |
| `mbf.crs` | Cross-section definition |
| `*.grd` | Computational grid |
| `*.enc` | Grid enclosure and boundary information |
| `*.dep` | Initial bed depth/elevation |

The long-basin case uses `domainl.grd`, `domainl.enc`, `domainlw.grd`, `domainlw.enc`, and `domainlw.dep`. The wide-basin case uses `domain.grd`, `domain.enc`, `domainw.grd`, `domainw.enc`, and `domainw.dep`.
