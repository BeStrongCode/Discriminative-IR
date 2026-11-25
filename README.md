# Learning a Strong Degradation Discrimination Capability for All-in-One Image Restoration

## 🔖 TODO

- [ ] Release testing code.
- [ ] Release pretrained weights.
- [ ] Release training code.

## ⚙️ Dependencies

- Python 3.8
- PyTorch 1.8.1
- CUDA 11.6

```bash
conda env create -f env.yml
conda activate disir
```

## 📦 Pretrained Weights

[Download pretrained weights from Baidu Netdisk](https://pan.baidu.com/s/12OAjzrr0Vbqa4b8Uve5aRA?pwd=maym). Use `maym` as the extraction code if prompted.

- `3D.ckpt` — 3D All-in-One Image Restoration
- `5D.ckpt` — 5D All-in-One Image Restoration

## 🧪 Test

Use `--ckpt-name` to specify the path to the checkpoint file and `--mode` to select the restoration task.

| `--mode` | Restoration task |
| --- | --- |
| `0` | Denoising |
| `1` | Deraining |
| `2` | Dehazing |
| `3` | 3D All-in-One (denoising, deraining, dehazing) |
| `4` | 5D All-in-One (denoising, deraining, dehazing, deblurring, low-light enhancement) |
| `5` | Real-world restoration (deraining, desnowing, dehazing) |
| `6` | Restoration on unseen degradations |

```bash
conda activate disir
python test.py --ckpt-name /path/to/3D.ckpt --mode 3
python test.py --ckpt-name /path/to/5D.ckpt --mode 4
```

For the 3D and 5D examples, set `--denoise_path`, `--derain_path`, and `--dehaze_path` to your local dataset paths. For 5D, also set `--deblur_path` and `--lowlight_path`.
