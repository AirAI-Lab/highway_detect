# BiResUNet++ / highway_detect（中文）

本仓库是用于道路裂缝 / 道路分割研究的代码与论文源文件集合，包含模型实现、训练/评估脚本与若干实验配置。项目核心包括 BiResUNet++、若干轻量化学生网络（如 LightSegNet）以及支持训练、推理、导出（ONNX/PyTorch）与实验复现的工具。

---

## 中文说明

### 主要内容
- `requirements.txt` — Python 依赖（训练/评估所需）
- `common/` — 网络实现、数据处理与工具函数（包括 BiResUNet 系列实现）
- `scripts/` 或 根目录脚本 — 包含训练、推理与评估的封装脚本
- `docs/` — 论文、图示与实验说明

### 快速开始（Windows PowerShell）
建议使用 Python 3.8 - 3.11 的虚拟环境。

1) 创建并激活虚拟环境（PowerShell）：

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2) 更新 pip 并安装依赖：

```powershell
python -m pip install -U pip
pip install -r requirements.txt
```

3) 运行训练或推理（示例）

仓库包含多个实验脚本，下面为示例命令；请以脚本实际参数与 README/脚本头部说明为准。

```powershell
# 示例（Linux 风格变量/后台运行示例在 Windows 上需要适配）
# 请在 Windows 上直接运行该 Python 命令，或使用 WSL/适配的后台管理方式。
python scripts/train_all_gpu.py --datasets mas,road --global-seed 1337

# 推理示例
python scripts/infer_student_checkpoint.py --checkpoint models/light_distill_best_ep30_miou0.7062.pth
```

注：仓库中有若干辅助脚本（例如用于数据预处理、HSV 统计的替换等），具体请查看相应 Python 文件和 `common/` 中的模块。

### 常见问题与提示
- 若使用 GPU，请确保已安装合适版本的 PyTorch，并能看到 CUDA 设备。
- 某些训练脚本在 Linux 下使用了 `nohup` / 环境变量导出语法，Windows PowerShell 上需作相应调整或使用 WSL。

---

## 联系 / 后续
如果需要我帮忙：自动化实验、制作 Dockerfile/WSL 指南、或生成可复现的训练配置，请开 issue 或回复本条说明具体任务与优先级。

祝你研究顺利！
