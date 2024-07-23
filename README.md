# YoloWGDinoArena

YOLOWorld vs GroundingDINO Arena

[HuggingFace Space](https://huggingface.co/spaces/pg56714/YoloWGDinoArena)

## Getting Started

### Installation

Use Anaconda to create a new environment and install the required packages.

setting CUDA_HOME
https://zhuanlan.zhihu.com/p/565649540s

```bash
# Create and activate a python 3.10 environment.
conda create -n yolowgdinoarena python=3.10 -y

conda activate yolowgdinoarena

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
