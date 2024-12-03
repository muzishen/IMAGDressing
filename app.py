import sys
from PIL import Image
import gradio as gr
import numpy as np
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from dressing_sd.pipelines.IMAGDressing_v1_pipeline_ipa_controlnet import IMAGDressing_v1
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torchvision import transforms
import cv2
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import diffusers

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from adapter.attention_processor import CacheAttnProcessor2_0, RefSAttnProcessor2_0, LoRAIPAttnProcessor2_0, RefLoraSAttnProcessor2_0
from diffusers import ControlNetModel, UNet2DConditionModel, \
    AutoencoderKL, DDIMScheduler
from adapter.resampler import Resampler

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL, UniPCMultistepScheduler
from typing import List

import torch

import argparse
import os

from controlnet_aux import OpenposeDetector
from insightface.app import FaceAnalysis
from insightface.utils import face_align


parser = argparse.ArgumentParser(description='IMAGDressing-v1')
parser.add_argument('--if_ipa', type=bool, default=True)
parser.add_argument('--if_control', type=bool, default=True)
parser.add_argument('--model_weight', type=str, required=True)
parser.add_argument('--server_port', type=int, required=True)
args = parser.parse_args()


args.device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(args.device).__contains__("cuda") else torch.float32

vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(dtype=torch.float16, device=args.device)
tokenizer = CLIPTokenizer.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="text_encoder").to(dtype=torch.float16, device=args.device)
image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder").to(dtype=torch.float16, device=args.device)
unet = UNet2DConditionModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet").to(dtype=torch.float16,device=args.device)

#face_model
app = FaceAnalysis(model_path="buffalo_l", providers=[('CUDAExecutionProvider', {"device_id": args.device})])
app.prepare(ctx_id=0, det_size=(640, 640))

# def ref proj weight
image_proj = Resampler(
    dim=unet.config.cross_attention_dim,
    depth=4,
    dim_head=64,
    heads=12,
    num_queries=16,
    embedding_dim=image_encoder.config.hidden_size,
    output_dim=unet.config.cross_attention_dim,
    ff_mult=4
)
image_proj = image_proj.to(dtype=torch.float16, device=args.device)

# set attention processor
attn_procs = {}
st = unet.state_dict()
for name in unet.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]

    if cross_attention_dim is None:
        attn_procs[name] = RefLoraSAttnProcessor2_0(name, hidden_size)
    else:
        attn_procs[name] = LoRAIPAttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

unet.set_attn_processor(attn_procs)
adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
adapter_modules = adapter_modules.to(dtype=torch.float16, device=args.device)
del st

ref_unet = UNet2DConditionModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet").to(
    dtype=torch.float16,
    device=args.device)
ref_unet.set_attn_processor(
    {name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})  # set cache


model_sd = torch.load(args.model_weight, map_location="cpu")["module"]

ref_unet_dict = {}
unet_dict = {}
image_proj_dict = {}
adapter_modules_dict = {}
for k in model_sd.keys():
    if k.startswith("ref_unet"):
        ref_unet_dict[k.replace("ref_unet.", "")] = model_sd[k]
    elif k.startswith("unet"):
        unet_dict[k.replace("unet.", "")] = model_sd[k]
    elif k.startswith("proj"):
        image_proj_dict[k.replace("proj.", "")] = model_sd[k]
    elif k.startswith("adapter_modules") and 'ref' in k:
        adapter_modules_dict[k.replace("adapter_modules.", "")] = model_sd[k]
    else:
        print(k)

ref_unet.load_state_dict(ref_unet_dict)
image_proj.load_state_dict(image_proj_dict)
adapter_modules.load_state_dict(adapter_modules_dict, strict=False)

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

control_net_openpose = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose",
    torch_dtype=torch.float16).to(device=args.device)

img_transform = transforms.Compose([
    transforms.Resize([640, 512], interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

openpose_model = OpenposeDetector.from_pretrained('lllyasviel/ControlNet').to(args.device)

unet.requires_grad_(False)
ref_unet.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)


def resize_img(input_image, max_side=640, min_side=512, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    ratio = min_side / min(h, w)
    w, h = round(ratio*w), round(ratio*h)
    ratio = max_side / max(h, w)
    input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
    w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)
    return input_image


def dress_process(garm_img, face_img, pose_img, prompt, cloth_guidance_scale, caption_guidance_scale,
                  face_guidance_scale, self_guidance_scale, cross_guidance_scale, if_ipa, if_postprocess,  if_control, denoise_steps, seed=42):
    image_face_fusion = pipeline('face_fusion_torch', model='damo/cv_unet_face_fusion_torch',model_revision='v1.0.0')
    if prompt is None:
        prompt = "a photography of a model"
    prompt = prompt + ', best quality, high quality'
    print(prompt, cloth_guidance_scale, if_ipa, if_control, denoise_steps, seed)
    clip_image_processor = CLIPImageProcessor()

    if not garm_img:
        raise gr.Error("请上传衣服 / Please upload garment")
    clothes_img = resize_img(garm_img)
    vae_clothes = img_transform(clothes_img).unsqueeze(0)
    ref_clip_image = clip_image_processor(images=clothes_img, return_tensors="pt").pixel_values

    if if_ipa:
        faces = app.get(face_img)
        if not faces:
            raise gr.Error("人脸检测异常，尝试其他肖像 / Abnormal face detection. Try another portrait")
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        face_image = face_align.norm_crop(face_img, landmark=faces[0].kps, image_size=224) # you can also segment the face
        

        face_clip_image = clip_image_processor(images=face_image, return_tensors="pt").pixel_values
    else:
        faceid_embeds = None
        face_clip_image = None

    if if_control:
        pose_img = openpose_model(pose_img.convert("RGB"))
        # pose_img.save('pose.png')
        pose_image = diffusers.utils.load_image(pose_img)
    else:
        pose_image = None

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    pipe = IMAGDressing_v1(unet=unet, reference_unet=ref_unet, vae=vae, tokenizer=tokenizer,
                            text_encoder=text_encoder, image_encoder=image_encoder,
                            ip_ckpt='./ckpt/ip-adapter-faceid-plus_sd15.bin',
                            ImgProj=image_proj, controlnet=control_net_openpose,
                            scheduler=noise_scheduler,
                            safety_checker=StableDiffusionSafetyChecker,
                            feature_extractor=CLIPImageProcessor)
    generator = torch.Generator(args.device).manual_seed(seed) if seed is not None else None
    output = pipe(
        ref_image=vae_clothes,
        prompt=prompt,
        ref_clip_image=ref_clip_image,
        pose_image=pose_image,
        face_clip_image=face_clip_image,
        faceid_embeds=faceid_embeds,
        null_prompt='',
        negative_prompt='bare, naked, nude, undressed, monochrome, lowres, bad anatomy, worst quality, low quality',
        width=512,
        height=640,
        num_images_per_prompt=1,
        guidance_scale=caption_guidance_scale,
        image_scale=cloth_guidance_scale,
        ipa_scale=face_guidance_scale,
        s_lora_scale= self_guidance_scale,
        c_lora_scale= cross_guidance_scale,
        generator=generator,
        num_inference_steps=denoise_steps,
    ).images
    
    if if_postprocess and if_ipa:

        output_array = np.array(output[0])

        bgr_array = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)

        bgr_image = Image.fromarray(bgr_array)
        result = image_face_fusion(dict(template=bgr_image, user=Image.fromarray(face_image.astype('uint8'))))
        return result[OutputKeys.OUTPUT_IMG]
    return output[0]


example_path = os.path.join(os.path.dirname(__file__), 'assets')

garm_list = os.listdir(os.path.join(example_path,"images"))
garm_list_path = [os.path.join(example_path,"garment",garm) for garm in garm_list]

face_list = os.listdir(os.path.join(example_path,"images"))
face_list_path = [os.path.join(example_path,"face",face) for face in face_list]

pose_list = os.listdir(os.path.join(example_path,"images"))
pose_list_path = [os.path.join(example_path,"pose",pose) for pose in pose_list]

def process_image(image):
    return image

image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## IMAGDressing-v1: Customizable Virtual Dressing 👕👔👚")
    gr.Markdown(
        "Customize your virtual look with ease—adjust your appearance, pose, and garment as you like<br>."
        "If you enjoy this project, please check out the [source codes](https://github.com/muzishen/IMAGDressing) and [model](https://huggingface.co/feishen29/IMAGDressing). Do not hesitate to give us a star. Thank you!<br>"
        "Your support fuels the development of new versions."
    )
    with gr.Row():
        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            example = gr.Examples(
                inputs=garm_img,
                fn=process_image,
                outputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path)

        with gr.Column():
            imgs = gr.Image(label="Face", sources='upload', type="numpy")
          
            with gr.Row():
                is_checked_face = gr.Checkbox(label="Yes", info="Use face ", value=False)
            example = gr.Examples(
                inputs=imgs,
                examples_per_page=10,
                fn=process_image,
                outputs=imgs,
                examples=face_list_path
            )
            with gr.Row():
                is_checked_postprocess = gr.Checkbox(label="Yes", info="Use postprocess ", value=False)

        with gr.Column():
            pose_img = gr.Image(label="Pose", sources='upload', type="pil")
            with gr.Row():
                is_checked_pose = gr.Checkbox(label="Yes", info="Use pose ", value=False)

            example = gr.Examples(
                inputs=pose_img,
                examples_per_page=8,
                fn=process_image,
                outputs=pose_img,
                examples=pose_list_path)


        with gr.Column():
            image_out = gr.Image(label="Output", elem_id="output-img", show_share_button=False)
    # Add usage tips below the output image
    gr.Markdown("""
    ### Usage Tips
    - **Upload Images**: Upload your desired garment, face, and pose images in the respective sections.
    - **Select Options**: Use the checkboxes to include face and pose in the generated output.
    - **View Output**: The resulting image will be displayed in the Output section.
    - **Examples**: Click on example images to quickly load and test different configurations.
    - **Advanced Settings**: Click on **Advanced Settings** to edit captions and adjust hyperparameters.
    - **Feedback**: If you have any issues or suggestions, please let us know through the [GitHub repository](https://github.com/muzishen/IMAGDressing).
    """)
    with gr.Column():
        try_button = gr.Button(value="Dressing")
        with gr.Accordion(label="Advanced Settings", open=True):
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(placeholder="Description of prompt ex) A beautiful woman dress Short Sleeve Round Neck T-shirts",value='A beautiful woman',
                                        show_label=False, elem_id="prompt")

            with gr.Row():
                cloth_guidance_scale = gr.Slider(label="Cloth guidance Scale", minimum=0.0, maximum=1.0, value=0.85, step=0.1,
                                             visible=True)
            with gr.Row():
                caption_guidance_scale = gr.Slider(label="Prompt Guidance Scale", minimum=1, maximum=10., value=6.5, step=0.1,
                                             visible=True)
            with gr.Row():
                face_guidance_scale = gr.Slider(label="Face Guidance Scale", minimum=0.0, maximum=2.0, value=0.9, step=0.1,
                                             visible=True)
            with gr.Row():
                self_guidance_scale = gr.Slider(label="Self-Attention Lora Scale", minimum=0.0, maximum=0.5, value=0.2, step=0.1,
                                             visible=True)
            with gr.Row():
                cross_guidance_scale = gr.Slider(label="Cross-Attention Lora Scale", minimum=0.0, maximum=0.5, value=0.2, step=0.1,
                                             visible=True)
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=50, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=20240508)

    try_button.click(fn=dress_process, inputs=[garm_img, imgs, pose_img, prompt, cloth_guidance_scale, caption_guidance_scale, face_guidance_scale, self_guidance_scale, cross_guidance_scale, is_checked_face, is_checked_postprocess, is_checked_pose, denoise_steps, seed],
                     outputs=[image_out], api_name='IMAGDressing-v1')

image_blocks.launch(server_port=args.server_port)  #
