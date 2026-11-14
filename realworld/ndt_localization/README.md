# NDT relocalization used in the paper

`reloc_seed.cpp` evaluates one relocalization per job: given an initial pose from a localization
arm, it registers the LiDAR scan recorded at the same instant against the prior map. All
cold-start numbers in the paper come from this program.

| Path | Contents |
|---|---|
| `reloc_seed.cpp` | offline relocalization of one job list |
| `build_reloc_seed.sh` | build script |
| `ndt_omp_ros2/` | pclomp (ROS 2 port of koide3/ndt_omp, BSD-2) |
| `lidar_localization_ros2/` | on-robot localization node (modified rsasaki0109/lidar_localization_ros2, BSD-2), set to the paper's settings below |

## Settings

| Stage | Setting |
|---|---|
| Scan preprocessing | keep 1 < r < 100 m (horizontal), then 0.2 m voxel grid; no deskewing |
| Coarse search | 1000 random points (`mt19937(2026)`), 4 iterations; yaw in 12° steps over ±60°; 5 m grid within 10 m, then a 3×3 grid at 2 m around the best 6 (208 alignments) |
| Fine alignment | best 16 coarse candidates, full scan, 35 iterations; selected by fitness only |
| Final alignment | one more alignment from the selected solution |
| NDT | resolution 1.0, step 0.1, ε 0.01, 4 threads, DIRECT7; target = full map |
| Success | converged and within 1 m of the pose logged while recording |

## Build and run

Tested on Ubuntu 22.04 with PCL 1.12.

```bash
bash realworld/ndt_localization/build_reloc_seed.sh /tmp/ndt_build
python realworld/make_reloc_jobs.py --arms raw:60,fol:60,star:60 --out /tmp/jobs.txt
/tmp/ndt_build/reloc_seed <map.pcd> <scan_dir> /tmp/jobs.txt /tmp/ndt.csv
python realworld/summarize_reloc.py --dir /tmp
```

Initial poses are written with six decimals by `make_reloc_jobs.py`; on failing arms the error
surface is flat, and a change of 1e-7 in the initial pose can lead to a different wrong minimum.
