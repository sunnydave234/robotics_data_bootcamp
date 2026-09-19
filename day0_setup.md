# Day 0 — Environment Setup

Get this done before Week 1 starts. All five steps should take under 2 hours total.

---

## 1 — Python Environment
*~30 min*

Use `pyenv` — not conda, not system Python. Conda's PyTorch on ARM has known MPS issues.

```bash
brew install pyenv
pyenv install 3.11.9
pyenv global 3.11.9
python -m venv ~/envs/robotics
source ~/envs/robotics/bin/activate
```

> **Note:** Add `source ~/envs/robotics/bin/activate` to your `.zshrc`. You'll forget to activate it on day 3.

---

## 2 — PyTorch with MPS
*~20 min*

```bash
pip install torch torchvision torchaudio
```

This single command installs the MPS-enabled build automatically. No special flags needed on M4.

Verify MPS is actually working — run this before anything else:

```python
import torch
assert torch.backends.mps.is_available(), 'MPS NOT available — stop and fix this'
assert torch.backends.mps.is_built(), 'MPS not built into this PyTorch install'
x = torch.randn(100, 100, device='mps')
print('MPS OK —', x.device)
```

> **M4 Mac — If MPS returns `False`:** Check macOS version (need 12.3+, you almost certainly have this). The failure is almost always a conda install that doesn't have MPS built in. If you used conda: delete that env, use `venv` + `pip`.

---

## 3 — Core Dependencies
*~30 min*

```bash
brew install hdf5 ffmpeg           # system deps first — h5py will fail without these
pip install h5py numpy pandas pyarrow
pip install opencv-python          # not opencv-contrib — you don't need it
pip install av                     # PyAV for FFmpeg bindings
pip install wandb dvc
pip install huggingface_hub datasets
```

> **M4 Mac — h5py Pitfall:** `h5py` on M4 sometimes ignores the brew HDF5. If `import h5py` fails:
> ```bash
> HDF5_DIR=$(brew --prefix hdf5) pip install h5py --no-binary=h5py
> ```
> This compiles from source against the correct library. Takes 2 minutes.

---

## 4 — LeRobot Install
*~30 min*

```bash
pip install lerobot
pip install gym-pusht              # simulation env for Month 2 — install now while you're here
```

LeRobot pulls ~30 dependencies. If it fails mid-install, check the error: it's usually `numba` or `gymnasium` on ARM.

**Fix:** `pip install lerobot --no-deps`, then install the LeRobot requirements file manually one by one.

Verify the install:

```bash
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; print('OK')"
```

---

## 5 — Accounts + GitHub Repo
*~20 min*

```bash
wandb login               # sign up at wandb.ai (free), paste API key when prompted
huggingface-cli login     # free account at huggingface.co, paste token
```

Create your GitHub portfolio repo:

1. Create repo: **`robotics-ml-portfolio`**
2. Add folder: `month-01-robot-data-forge/`
3. Push an empty `README.md`

This repo is now live — keep it that way.
