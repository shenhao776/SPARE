# Physical-robot experiment

Code for the cold-start relocalization experiment on CustomCampus (821 trials) and the
photo-to-goal demonstration. The recorded sensor data, prior map and per-trial results are not
included.

## Online service

`star_server.py` serves the deployed pipeline (frozen retriever + SPARE head) over a socket and an
optional phone web interface:

```bash
DATASET=customcampus python realworld/star_server.py --mode star --with-webapp
```

`--mode raw` serves the retriever alone, for comparison.

## Cold-start trials

Each trial takes one telephoto query frame, localizes it with an arm (raw retrieval, FoL, or
SPARE), and uses the selected pose to initialize NDT relocalization of the LiDAR scan recorded at
the same instant. Arms differ only in the initial pose.

| Step | Script |
|---|---|
| export query frames, LiDAR scans and logged poses from the recording | `export_reloc_scans.py` |
| localize every query with an arm | `run_reloc_trials.py`, `add_baseline_arm.py` |
| write the NDT job list | `make_reloc_jobs.py` |
| relocalize | `ndt_localization/reloc_seed.cpp` (see `ndt_localization/README.md`) |
| summarize (success, conditional rates, paired comparison) | `summarize_reloc.py` |
| figures | `render_reloc_overlay.py`, `build_fig5_assets.py` |

`rerun_star_arm.sh` re-runs only the SPARE arm with another head, reusing the raw and FoL arms,
which do not depend on the head.
