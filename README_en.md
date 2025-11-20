# BiResUNet++ / highway_detect (English)

This repository contains code and LaTeX sources for research on road segmentation (e.g., crack detection) from UAV imagery. The main artifacts include the BiResUNet++ model family, lightweight student networks (e.g., LightSegNet), and scripts/tools for training, evaluation, and deployment (PyTorch/ONNX).

---

## Overview

### Contents (important files)
- `requirements.txt` — Python dependencies for training/evaluation.
- `common/` — model implementations, data-processing utilities and helper modules.
- `scripts/` and root-level scripts — entry points for experiments, training and inference.

### Quick start (Windows PowerShell)
We recommend using Python 3.8–3.11 inside a virtual environment.

1) Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2) Upgrade pip and install dependencies:

```powershell
python -m pip install -U pip
pip install -r requirements.txt
```

3) Run training or inference (examples)

There are multiple experiment scripts in the repo. Below are basic examples — adapt flags/parameters to your needs.

```powershell
# Training example
python scripts/train_all_gpu.py --datasets mas,road --global-seed 1337

# Inference example
python scripts/infer_student_checkpoint.py --checkpoint models/light_distill_best_ep30_miou0.7062.pth
```

Notes:
- Some scripts assume a Unix-like environment for process backgrounding or environment variables (e.g., `nohup`, `export`). On Windows use equivalent PowerShell commands or WSL.
- Ensure the correct PyTorch + CUDA versions are installed if you want GPU training.

---

## Contact / Next steps
If you want help automating experiments, preparing a Dockerfile/WSL guide, or creating reproducible training configs, please open an issue or reply with the specific task and priority.

Good luck with your experiments!
