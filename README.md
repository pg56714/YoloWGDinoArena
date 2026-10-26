# YoloWGDinoArena

> **Security:** see [SECURITY.md](SECURITY.md) for known dependency issues.

YOLOWorld vs GroundingDINO Arena

[HuggingFace Space](https://huggingface.co/spaces/pg56714/YoloWGDinoArena)

## Getting Started

### Installation

Use Anaconda to create a new environment and install the required packages.

setting CUDA_HOME
https://zhuanlan.zhihu.com/p/565649540s

```bash
# Create and activate a python 3.12 environment.
conda create -n yolowgdinoarena python=3.12 -y

conda activate yolowgdinoarena

pip install -r requirements.txt

pip install -e .
```

create weights folder

cd weights

download https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

put the downloaded file in the weights folder

### Running the Project

```bash
python app.py
```
