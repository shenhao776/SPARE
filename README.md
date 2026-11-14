# SPARE: Self-Supervised Position-Annotated Re-ranking for Single-Image Constrained-FOV Localization

Code and pre-trained weights for the paper of the same title.

SPARE re-ranks the Top-K shortlist of a frozen visual place recognition (VPR) retriever when the
query image has a narrower field of view (FOV) than the mapping camera, e.g. a phone at optical
zoom or a robot camera that differs from the mapping rig. It is trained from the geo-tags of the
database alone: no manual labels, no real queries, and no change to the retriever.

**Downloads:** [pre-trained weights](https://drive.google.com/file/d/1ZnlSc6X2S2DfQgIO2Hfq6owfRY10-KVV/view) ·
[all released assets (Google Drive)](https://drive.google.com/drive/folders/147V5h46YrM6vjPtyNomhvaX8UB2tF2vx) ·
[GitHub release](https://github.com/shenhao776/SPARE/releases/tag/v1.0)

## Method

Synthetic crops of database images serve as constrained-FOV pseudo-queries whose positions are
known; retrieving them returns a shortlist, and the database positions label every candidate. For
every query–candidate pair, SPARE builds a compact state of seven inputs: six pair channels in
two groups plus the query's magnification:

| Group | Inputs |
|---|---|
| Match geometry (4) | SuperPoint + LightGlue inlier count; the areas of the inlier convex hulls on the database and query images, and their ratio |
| Patch appearance (2) | patch agreement under DINOv2, and under MegaLoc's place-recognition-tuned DINOv2 backbone |
| Query input (1) | the query's magnification (the inverse crop ratio for synthetic crops, the optical-zoom factor for real telephoto queries) |

A small MLP (7→64→64→1) scores each candidate and a softmax over the K candidates gives the
re-ranking. The state never reads the retriever's score or rank. Because neither the state nor
the labels refer to a particular place, the pseudo-queries of several maps are pooled: one
*universal head*, trained once on seven mapping runs, is used for every dataset, success radius
and retriever in the paper.

## Results

Top-1 recall with a MegaLoc Top-10 shortlist; SPARE is the universal head (six-seed mean).

| Setting | Raw retrieval | + SPARE |
|---|---|---|
| Five trajectory datasets (RobotCar, 4Seasons, NCLT, Boreas, CustomCampus), mean at 5 / 10 / 15 m | 24.9 / 42.4 / 54.6 % | 52.3 / 70.4 / 77.3 % |
| Pitts30k, constrained-FOV queries, 25 m (zero-shot) | 61.4 % | 67.9 % |
| MSLS, constrained-FOV queries, 25 m (zero-shot) | 48.2 % | 57.4 % |
| Physical robot, cold-start LiDAR relocalization success, 821 trials | 57.9 % | 78.9 % |

Re-ranking ten candidates takes 0.58 s per query on an RTX 4070.

## Pre-trained weights and assets

Everything is in one [Google Drive folder](https://drive.google.com/drive/folders/147V5h46YrM6vjPtyNomhvaX8UB2tF2vx). Each archive is extracted in the
repository root and unpacks under `assets/`.

| File | Size | Contents |
|---|---|---|
| [`spare_weights.tar.gz`](https://drive.google.com/file/d/1ZnlSc6X2S2DfQgIO2Hfq6owfRY10-KVV/view) | 3 MB | Every released head, six seeds each (30 heads): the universal head, the mixed-crop universal head of the field-of-view table, the ablation variants, per-map and leave-one-dataset-out heads, and radius-matched heads. Each `.pth` has a `.meta.json` with its training settings. Also attached to the [GitHub release](https://github.com/shenhao776/SPARE/releases/tag/v1.0). |
| [`spare_release_assets.tar.gz`](https://drive.google.com/file/d/1HER3zPUIg9hHhAL1goQxmLLyv6EZ9lv5/view) | 46 MB | State caches, heads and same-pool baseline outputs for the RobotCar checks (level 1 below). Also attached to the [GitHub release](https://github.com/shenhao776/SPARE/releases/tag/v1.0). |
| [`spare_replay_assets.tar.gz`](https://drive.google.com/file/d/1WjbSJXdaHv45gE0dF2XJnkaGEkWz7nHo/view) | 965 MB | Everything level 2 below reads: test-split state caches for nine retrievers and for the field-of-view crops, same-pool baseline outputs and all heads for RobotCar, 4Seasons, NCLT, Boreas, CustomCampus, Pitts30k and MSLS, the recorded timing run, and the MegaLoc descriptors of uncropped Pitts30k and MSLS. Includes the two archives above. |

SHA-256 checksums are in [`SHA256SUMS.txt`](https://drive.google.com/file/d/1va36-GlqtXnvAFK8yjSgADofXNDAcreo/view).

The universal head used throughout the paper is

```
assets/MegaLoc_8448/models/vpr_reranking_gru_arch=single_frame_deriv=False_prog=False_area=True_dino=True_zoom=True_vprpatch=True_innorm=False_UnionMaps_no_roi.pth
```

(seed 2026; `_seed1` … `_seed5` are the other five seeds, and `UnionMapsMix` is the mixed-crop
head). The heads are about 4.7k parameters each. The frozen retrievers and the evidence
extractors (SuperPoint, LightGlue, DINOv2) are public models and are not redistributed here;
`tools/prepare_model_checkpoints.py` fetches the retriever checkpoints from their upstream
releases. To run the online pipeline
(`realworld/star_server.py`) on a map that has no head of its own, point it at the universal
head with the `STAR_RERANK_WEIGHT` environment variable.

**CustomCampus.** The raw CustomCampus recordings (our own images) are not released. Its processed
evaluation assets (state caches, same-pool baseline outputs and heads) are, so every CustomCampus
number in the paper's tables can be replayed.

## Installation

```bash
conda env create -f env/clean_env.yml   # GPU environment
conda activate vpr_env
```

See `env/readme.md` for the CPU-only replay environment.

## Reproducing the paper

Both levels below run from the released assets.

**1. RobotCar spot check (CPU, a few minutes).** Extract `spare_release_assets.tar.gz` in the
repository root, then:

```bash
bash quick_reproduce.sh      # main table, RobotCar column
bash robotcar_reproduce.sh   # RobotCar slice of the released tables: main table, cross-retriever, 25 m, field of view
```

Both scripts re-evaluate the released state caches with the released heads (six seeds) and every
same-pool baseline output, and check each number against the tables in `results/tables/`.

**2. Every paper table from cached assets (one GPU, about 45 minutes).** Extract
`spare_replay_assets.tar.gz` in the repository root, then:

```bash
ENABLE_SCENE_MODELS_CACHE=True python run_paper_tables.py
bash compare_results.sh
python extra_reproduce.py
```

Add `--cpu` to the two Python commands to run without a GPU. `run_paper_tables.py` prints its
tables, saves them to `logs/results/paper_tables_replay_<time>.log` and writes CSVs to
`results/replay/`; `compare_results.sh` compares those CSVs with the published ones in
`results/tables/`; `extra_reproduce.py` checks every number it replays against the published
values. Both exit non-zero if a number differs.

| Paper table | Command |
|---|---|
| Tables 1-4 and 6 | `run_paper_tables.py`, checked by `compare_results.sh` |
| Table 6, rank-fusion row | `extra_reproduce.py --parts fusion` |
| Table 5 (field of view) | `extra_reproduce.py --parts fov` |
| Table 7 (latency), from the logs of the timing run | `extra_reproduce.py --parts latency` |

`extra_reproduce.py` also replays numbers quoted in the text: the field-of-view channel ablation
of Sec. V-E (`--parts channels`), MegaLoc under the standard evaluation of Sec. I
(`--parts standard`). Table 7 is replayed from the recorded timing run.

**Fine-tuned baselines.** The rows marked † in Tables 1, 2 and 5 fine-tune Pair-VPR (stage 2),
SelaVPR and GeoWarp from their released weights on the same pseudo-queries as SPARE, each with
its official training code (cloned into `3rdparty/`) driven by `tools/pairvpr_target/`,
`tools/selavpr_target/` and `tools/geowarp_target/`. Their same-pool re-ranking outputs are in the replay
assets, so the † rows replay with level 2; the fine-tuned weights are not redistributed.

## Repository layout

| Path | Contents |
|---|---|
| `reproduce.py` | the full pipeline used for the paper (cache building, head and baseline training) |
| `run_paper_tables.py`, `extra_reproduce.py` | replay the paper results from built assets |
| `train_agent.py`, `tools/train_union_head.py` | head training (called by `reproduce.py`) |
| `core/`, `sl_agent/` | retrieval, evidence extraction, state construction, head |
| `configs/` | protocol (`default.yaml`), datasets, retrievers, paper tables |
| `tools/` | cache building, channel injection, baselines on the shared pool, timing |
| `3rdparty_compat/` | compatibility patches for third-party baselines (cloned under `3rdparty/`) |
| `realworld/` | physical-robot cold-start experiment (online service, NDT relocalization); code only |
| `results/tables/` | the published numbers that the reproduction scripts check against |
| `tests/` | unit tests (`python -m unittest discover -s tests`) |

## License

MIT (see `LICENSE`). Third-party code under `realworld/ndt_localization/` keeps its own license.
