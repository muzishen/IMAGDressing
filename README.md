# **IMAGDressing: Interactive Modular Apparel Generation for Dressing**
## IMAGDressing-v1: Customizable Virtual Dressing

<a href='https://imagdressing.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://imagdressing.github.io/'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/feishen29/IMAGDressing'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
[![GitHub stars](https://img.shields.io/github/stars/muzishen/IMAGDressing?style=social)](https://github.com/muzishen/IMAGDressing/stargazers)


### 🚀 **Key Features:**
1. **Simple Architecture**: IMAGDressing-v1 can generating high garment fidelity and allow for user-controlled scene editings. 
2. **Flexible Plugin Compatibility**: IMAGDressing-v1 modestly integrates with extension plugins such as [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [ControlNet](https://github.com/lllyasviel/ControlNet), [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter), and [AnimateDiff](https://github.com/guoyww/AnimateDiff).
3. **Rapid Customization**: Enables rapid customization in seconds without the need for additional LoRA training.



---
## 🔥 **Examples**



## Release
- [2024/05/28] 🔥 We release the inference code of SD1.5 that is compatible with [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) and [ControlNet](https://github.com/lllyasviel/ControlNet).
- [2024/05/30] 🔥 We release the [Gradio_demo](https://sf.dictdoc.site/) of IMAGDressing-v1.
- [2024/05/08]  🔥 We launch the [project page](https://imagdressing.github.io/) of IMAGDressing-v1.

## Introduction

To address the need for flexible and controllable customizations in virtual try-on systems, we propose IMAGDressing-v1. Specifically, we introduce a garment UNet that captures semantic features from CLIP and texture features from VAE. Our hybrid attention module includes a frozen self-attention and a trainable cross-attention, integrating these features into a frozen denoising UNet to ensure user-controlled editing. We will release a comprehensive dataset, IGv1, with over 200,000 pairs of clothing and dressed images, and establish a standard data assembly pipeline. Furthermore, IMAGDressing-v1 can be combined with extensions like ControlNet, IP-Adapter, T2I-Adapter, and AnimateDiff to enhance diversity and controllability. 

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
```

---
## 🎉 How to Use

### <span style="color:red">Important Reminder</span>


### 1. Random faces and poses to dress the assigned clothes 

```sh
python inference.py --cloth_path [your cloth path]
```


### 2. Random faces use a given pose to dress a given outfit 

```sh
python inference.py --cloth_path [your cloth path] --face_path [your face path]
```

### 3. Specify the face and posture to wear the specified clothes

```sh
python inference.py --cloth_path [your cloth path] --face_path [your face path] --pose_path [your posture path]
```



## Get Involved
Join us on this exciting journey to transform virtual try-on systems. Star⭐️ our repository to stay updated with the latest advancements, and contribute to making **IMAGDressing** the leading solution for virtual clothing generation.


## Acknowledgement
xxx

