# **_IMAGDressing : Customizable Virtual Dressing_**

<a href='https://imagdressing.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://imagdressing.github.io/'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/feishen29/IMAGDressing'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>


### 🚀 **Key Features:**

1. Image generation boasts exceptionally **high garment fidelity** and supports **scene editing**.
2. Functions as an adapter, **compatible with other foundational models** in the community such as [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) and [ControlNet](https://github.com/lllyasviel/ControlNet).
3. Enables **rapid customization** in seconds without the need for additional LoRA training.


---
## 🔥 **Examples**



## Release
- [2024/05/28] 🔥 We release the inference code of SD1.5 that is compatible with [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) and [ControlNet](https://github.com/lllyasviel/ControlNet).
- [2024/05/30] 🔥 We release the [Gradio_demo](https://sf.dictdoc.site/).
- [2024/05/08]  🔥 We launch the [project page](https://imagdressing.github.io/).

## Introduction

xxxx

![framework](assets/pipeline.png)

## 🔧 Requirements

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.0](https://pytorch.org/)
- cuda==11.8

```bash
conda create --name IMAGDressing python=3.8.10
conda activate IMAGDressing
pip install -U pip

# Install requirements
pip install -r requirements.txt
