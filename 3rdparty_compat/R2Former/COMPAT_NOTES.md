# R2Former compatibility changes

Upstream: <https://github.com/bytedance/R2Former> (Apache-2.0, see `LICENSE`).
The method is unchanged; `apply_r2former.sh` copies these files over a clone in
`3rdparty/R2Former`. Only the released weights are evaluated in the paper.

| File | Change | Reason |
|---|---|---|
| `util.py` | `import torchscan` made optional | used only by `get_flops()`; the package pins `torch<2.0` |
| `datasets_ws.py` | `np.float` -> `float` (2 places) | the alias was removed in NumPy 1.24; identical semantics |
| `datasets_ws.py` | ragged positive lists filtered into explicit object arrays | NumPy 1.24 no longer converts ragged sequences in `np.delete`; identical result |
| `model/network.py` | `ViTModel` imported only when a ViT backbone is selected | the top-level import pulls in an incompatible torchaudio build |
| `model/network.py` | Google Drive downloader imported under either package name, optional | the package was renamed in version 1.x; used only to download weights |
