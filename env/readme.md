# Environments

## Full GPU reproduction

This is the environment for every `reproduce.py` command, including MegaLoc
feature extraction and SuperPoint + LightGlue:

```bash
conda env create -f env/clean_env.yml
conda activate vpr_env
```

For an existing environment:

```bash
conda env update -n vpr_env -f env/clean_env.yml --prune
```

LightGlue is installed from a pinned upstream Git revision by the YAML. A
machine-local editable checkout is intentionally not used, so the environment
contract is the same on every device. `reproduce.py` imports both `SuperPoint`
and `LightGlue` before starting any stage and exits immediately if unavailable.
`gdown` is included so full paper reproduction can materialize the public
MixVPR/SuperVLAD checkpoints automatically. For an offline machine, point
`STAR_VPR_PRETRAINED_SOURCE` at an assets backup with the same directory layout.

## CPU assets-only table replay

`eva_env.yml` is intentionally smaller. It is only for replaying already-built
trajectory JSONs and agent weights; it cannot run `reproduce.py`:

```bash
conda env create -f env/eva_env.yml
conda activate star_vpr_eval
python run_paper_tables.py --cpu
```

Missing table assets are rendered as `-` by `run_paper_tables.py`.
