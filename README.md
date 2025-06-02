# **👔IMAGDressing👔: Interactive Modular Apparel Generation for Virtual Dressing**



## 📦️ Release
- [2025/05/30]  🔥 The supplementary materials for IMAGDressing-v1 are available [here](https://github.com/muzishen/IMAGDressing/blob/main/assets/AAAI2025_IMAGDressing_v1_supp.pdf).
- [2024/12/10] 🔥 IMAGDressing-v1 is accepted by AAAI 2025.
- [2024/08/24] 🔥 We add the [train code](https://github.com/muzishen/IMAGDressing/blob/main/train.py), feel free to give it a try!
- [2024/08/23] 🔥 We release the [IGPair dataset](https://huggingface.co/datasets/IMAGDressing/IGPair) publicly available for download.
- [2024/07/30] 🔥 We release the [WebUI Code](https://github.com/muzishen/IMAGDressing/blob/main/app.py) for gradio interface.
- [2024/07/26] 🔥 We release the [online webui]( https://huggingface.co/spaces/feishen29/IMAGDressing-v1)，thanks to [ZeroGPU](https://huggingface.co/zero-gpu-explorers) for providing free A100 GPUs. And the original [Gradio_demo](https://sf.dictdoc.site/)  is soon to be deprecated. 
- [2024/07/19] 🔥 We release the [code](https://github.com/muzishen/IMAGDressing/blob/main/inference_IMAGdressing_cartoon_style.py) and [examples](https://github.com/muzishen/IMAGDressing/blob/main/assets/cartoon.png) for cartoon-style virtual dressing.
- [2024/07/18] 🔥 We release the [technical report of IMAGDressing-v1](http://arxiv.org/abs/2407.12705) and [CAMI metric code](https://github.com/muzishen/IMAGDressing/tree/main/metric).
- [2024/07/16] 🔥 We add the batch inference for [full VD](https://github.com/muzishen/IMAGDressing/blob/main/inference_IMAGdressing_controlnetinpainting_folder.py) and [VTON](https://github.com/muzishen/IMAGDressing/blob/main/inference_IMAGdressing_ipa_controlnetpose_folder.py). Thanks @ZhaoChaoqun for the contribution.
- [2024/07/01] 🔥 We release the test cases in the assets/images directory.
- [2024/06/21] 🔥 We release the inpainting feature to enable outfit changing. **[Experimental Feature](assets/inpainting.png)**.
- [2024/06/13] 🔥 We release the [Gradio_demo](https://sf.dictdoc.site/) of IMAGDressing-v1 (**Service deprecation imminent**).
- [2024/05/28] 🔥 We release the inference code of SD1.5 that is compatible with [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) and [ControlNet](https://github.com/lllyasviel/ControlNet).
- [2024/05/08]  🔥 We launch the [project page](https://imagdressing.github.io/) of IMAGDressing-v1.

---

## IMAGDressing-v1: Customizable Virtual Dressing
<a href='https://imagdressing.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='http://arxiv.org/abs/2407.12705'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/feishen29/IMAGDressing'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/datasets/IMAGDressing/IGPair'><img src='https://img.shields.io/badge/Dataset-IGPair-orange'></a>
[![GitHub stars](https://img.shields.io/github/stars/muzishen/IMAGDressing?style=social)](https://github.com/muzishen/IMAGDressing)



### 🚀 **Key Features:**
1. **Simple Architecture**: IMAGDressing-v1 generates lifelike garments and facilitates easy user-driven scene editing.
2. **New Task, Metric, and Dataset**: Introduces the virtual dressing (VD) task, designs a comprehensive affinity metric index (CAMI), and releases the IGPair dataset.
3. **Flexible Plugin Compatibility**: Seamlessly integrates with extension plugins such as [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [ControlNet](https://github.com/lllyasviel/ControlNet), [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter), and [AnimateDiff](https://github.com/guoyww/AnimateDiff).
4. **Rapid Customization**: Allows for rapid customization within seconds without the need for additional LoRA training.



## 🔥 **Dataset Demo**
You can download the dataset from [Baidu Cloud](https://pan.baidu.com/s/1y-VnKWgHHCPV5M4zQUmn7g?pwd=IMAG) or [Huggingface Dataset](https://huggingface.co/datasets/IMAGDressing/IGPair). By requesting access, you agree to use the data only for academic and personal purposes and not for commercial use.

![Dataset Demo](assets/dataset_demo.png)

## 🔥 **Examples**

<div style="display: flex; justify-content: space-around;">
  <img src="assets/scrolling_images1.gif" alt="GIF 1" width="200" />
  <img src="assets/scrolling_images2.gif" alt="GIF 2" width="200" />
  <img src="assets/scrolling_images3.gif" alt="GIF 3" width="200" />
    <img src="assets/scrolling_images4.gif" alt="GIF 4" width="200" />
</div>


![compare](assets/compare_magic2.png)

### <span style="color:red">Conbined with IP-Adapter and Controlnet-Pose</span>

![compare](assets/pose.png)

![compare](assets/person.png)

### <span style="color:red">Support text prompts for different scenes</span>
![different scenes](assets/different_scenes.png)

### <span style="color:red">Supports outfit changing in specified areas (Experimental Feature)</span>
![inpainting](assets/inpainting.png)

### <span style="color:red">Supports generating cartoon-style images (Experimental Feature)</span>
![cartoon](assets/cartoon.png)


## 🏷️  Introduction

To address the need for flexible and controllable customizations in virtual try-on systems, we propose IMAGDressing-v1. Specifically, we introduce a garment UNet that captures semantic features from CLIP and texture features from VAE. Our hybrid attention module includes a frozen self-attention and a trainable cross-attention, integrating these features into a frozen denoising UNet to ensure user-controlled editing. We will release a comprehensive dataset, IGPair, with over 300,000 pairs of clothing and dressed images, and establish a standard data assembly pipeline. Furthermore, IMAGDressing-v1 can be combined with extensions like ControlNet, IP-Adapter, T2I-Adapter, and AnimateDiff to enhance diversity and controllability. 

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


## 🌐 Download Models
You can download our models from [HuggingFace](https://huggingface.co/feishen29/IMAGDressing) or [百度云](https://pan.baidu.com/s/1-69aFUjtrsGnD0OlU0dMzQ?pwd=imag).  You can download the other component models from the original repository, as follows.
- [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse).
- if train: [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), if test: [SG161222/Realistic_Vision_V4.0_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V4.0_noVAE).
- [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID).
- [lllyasviel/control_v11p_sd15_openpose](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose).

## 🎉 How to Train
```sh
# Please download the IGPair data first and modify the path in run.sh
sh run.sh
```

## 🎉 How to Test

### <span style="color:red">Important Reminder</span>


### 1. Random faces and poses to dress the assigned clothes 

```sh
python inference_IMAGdressing.py --cloth_path [your cloth path]
```


### 2. Random faces use a given pose to dress a given outfit 

```sh
python inference_IMAGdressing_controlnetpose.py --cloth_path [your cloth path] --pose_path [your posture path]
```

### 3. Specify the face and posture to wear the specified clothes

```sh
python inference_IMAGdressing_ipa_controlnetpose.py --cloth_path [your cloth path] --face_path [your face path] --pose_path [your posture path]
```

### 4. Specify the model to wear the specified clothes (Experimental Feature)
<span style="color:red">Please download the humanparsing and openpose model file from [IDM-VTON-Huggingface](https://huggingface.co/spaces/yisol/IDM-VTON/tree/main) to the ckpt folder first.</span>
```sh
python inference_IMAGdressing_controlnetinpainting.py --cloth_path [your cloth path] --model_path [your model path]
```
### 5. Specify the carton style for generate images (Experimental Feature)
```sh
python inference_IMAGdressing_counterfeit-v30.py --cloth_path [your cloth path] --model_path [your model path]
```

## 🎉 How to Eval
### Evaluating the CAMI-U score
Please use our inference_IMAGdressing.py to generate model images. Then, generate the cloth_mask based on the model image. You can use the [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) to generate cloth mask.
Finally, use the following code to evaluate the score for image generation without specified pose, face, and text scenarios.
```sh
python metric/eval.py
```

### Evaluating the CAMI-S score 
First, use inference_IMAGdressing_ipa_controlnetpose.py to generate model images. Then, generate the cloth mask based on the model image. 
Finally, use the following code to evaluate the image generation score for specified pose, face, and text scenarios.
```sh
python metric/eval_s.py
```


## 🤗Gradio interface 🤗

We also provide a Gradio <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> interface for a better experience, just run by:


```bash
pip install modelscope==1.15.0
pip install mmcv-full==1.7.2
pip install mmdet==2.26.0

python app.py  --model_weight $MODEL PATH --server_port  7860
```

You can specify the `--server_port` arguments to satisfy your needs!


**Or, try it out effortlessly on [HuggingFace](https://huggingface.co/spaces/feishen29/IMAGDressing-v1) 🤗**


## 📚 Get Involved
Join us on this exciting journey to transform virtual dressing systems. Star⭐️ our repository to stay updated with the latest advancements, and contribute to making **IMAGDressing** the leading solution for virtual dressing generation.
### Third-party Implementations of IMAGDressing-v1:
- [windows-webui](https://github.com/v3ucn/IMAGdressing_WebUi_For_Windows) by @v3ucn
- [comfyUI](https://github.com/AIFSH/IMAGDressing-ComfyUI) by @AIFSH


## Acknowledgement
We would like to thank the contributors to the [IDM-VTON](https://github.com/yisol/IDM-VTON), [MagicClothing](https://github.com/ShineChen1024/MagicClothing), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [ControlNet](https://github.com/lllyasviel/ControlNet), [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter), and [AnimateDiff](https://github.com/guoyww/AnimateDiff) repositories, for their open research and exploration.

The IMAGDressing code is available for both academic and commercial use. However, the models available for manual and automatic download from IMAGDressing are intended solely for non-commercial research purposes. Similarly, our released checkpoints are restricted to research use only. Users are free to create images using this tool, but they must adhere to local laws and use it responsibly. The developers disclaim any liability for potential misuse by users.

## 📝 Citation

If you find IMAGDressing-v1 useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{shen2024IMAGDressing-v1,
  title={IMAGDressing-v1: Customizable Virtual Dressing},
  author={Shen, Fei and Jiang, Xin and He, Xin and Ye, Hu and Wang, Cong, and Du, Xiaoyu, Li Zechao, and Tang, Jinhui},
  booktitle={arXiv preprint arXiv:2407.12705},
  year={2024}
}
```

## 🕒 TODO List
- [x] Gradio demo
- [x] Inference code
- [x] Model weights (512 sized version)
- [x] Support inpaint
- [ ] Model weights (More higher sized version)
- [x] Paper
- [x] Evaluate metric code
- [x] IGPair dataset
- [x] Training code
- [ ] Video Dressing
- [ ] Others, such as User-Needed Requirements


## 👉 **My other projects:**  
- [IMAGGarment-1](https://github.com/muzishen/IMAGGarment-1): Fine-grained controllable garment generation.  
- [IMAGHarmony](https://github.com/muzishen/IMAGHarmony): Controllable image editing with consistent object layout.  
- [IMAGPose](https://github.com/muzishen/IMAGPose): Pose-guided person generation with high fidelity.  
- [RCDMs](https://github.com/muzishen/RCDMs): Rich-contextual conditional diffusion for story visualization.  
- [PCDMs](https://github.com/tencent-ailab/PCDMs): Progressive conditional diffusion for pose-guided image synthesis.  

## 📨 Contact
If you have any questions, please feel free to contact with me at shenfei140721@126.com.
