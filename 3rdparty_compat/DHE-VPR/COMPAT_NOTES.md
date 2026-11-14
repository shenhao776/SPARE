# DHE-VPR compatibility changes

Upstream: <https://github.com/Lu-Feng/DHE-VPR>, commit `5950430` (MIT, see `LICENSE`).
The method is unchanged; `apply_dhe.sh` copies these files over a clone in
`3rdparty/DHE-VPR`. The paper reports the released weights only.

| File | Change | Reason |
|---|---|---|
| `network.py` | removed the unused `from transformers import DeiTModel` | the import pulls in an incompatible torchaudio build |
| `network.py` | `cct_14_7x2_384(pretrained=True)` -> `False` | the ImageNet download host fails certificate checks; `--resume_fe` overwrites these weights anyway (strict load) |
| `dataset_geoloc.py`, `datasets_ws.py` | `np.float` -> `float` | the alias was removed in NumPy 1.24; identical semantics |
| `util.py` | removed NumPy aliases (`np.float`, `np.int`, ...) -> builtins | same reason |
| `datasets_util.py` | images are read with `.convert("RGB")` | some 4Seasons frames are single-channel |
| `dataset_qp.py` | dataset names other than `msls` take the coordinate-based branch | the upstream code only handles `msls` and `pitts30k` by name |
| `train_dhe.py` | a checkpoint is saved every epoch | upstream saves every tenth epoch only |

Released feature extractor (MSLS, 118 MB), stored as `assets/dhe_vpr/CCT14_msls.pth`:

```bash
curl -sL 'https://drive.usercontent.google.com/download?id=1kXGixMjU_imBu0Yq5cZVovBUx3MGyoZ8&export=download&confirm=t' -o CCT14_msls.pth
```
