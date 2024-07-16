from dressing_sd.pipelines.IMAGDressing_v1_pipeline_controlnet_inpainting import IMAGDressing_v1
import os
import torch

from PIL import Image
from diffusers import ControlNetModel, UNet2DConditionModel, \
    AutoencoderKL, DDIMScheduler
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from adapter.attention_processor import CacheAttnProcessor2_0, RefSAttnProcessor2_0, CAttnProcessor2_0
import argparse
from adapter.resampler import Resampler
import numpy as np
from diffusers.utils import load_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from preprocess.utils_mask import get_mask_location
import cv2 as cv


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def resize_img(input_image, max_side=640, min_side=512, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / max(h, w)
    input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
    w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    return input_image


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0

    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    # image[image_mask > 0.5] = 0  # set as masked pixel
    # cv.imwrite("control_image.jpg", np.array(image * 255))
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def prepare(args):
    generator = torch.Generator(device=args.device).manual_seed(42)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16, device=args.device)
    tokenizer = CLIPTokenizer.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="text_encoder").to(
        dtype=torch.float16, device=args.device)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder").to(
        dtype=torch.float16, device=args.device)
    unet = UNet2DConditionModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet").to(
        dtype=torch.float16,
        device=args.device)

    # load ipa weight
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
            attn_procs[name] = RefSAttnProcessor2_0(name, hidden_size)
        else:
            attn_procs[name] = CAttnProcessor2_0(name, hidden_size=hidden_size,
                                                 cross_attention_dim=cross_attention_dim)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules = adapter_modules.to(dtype=torch.float16, device=args.device)
    del st

    ref_unet = UNet2DConditionModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet").to(
        dtype=torch.float16,
        device=args.device)
    ref_unet.set_attn_processor(
        {name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})  # set cache

    # weights load
    model_sd = torch.load(args.model_ckpt, map_location="cpu")["module"]

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
    control_net = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint",
                                                  torch_dtype=torch.float16).to(device=args.device)
    pipe = IMAGDressing_v1(unet=unet, reference_unet=ref_unet, vae=vae, tokenizer=tokenizer,
                           text_encoder=text_encoder, image_encoder=image_encoder,
                           ImgProj=image_proj,
                           scheduler=noise_scheduler,
                           controlnet=control_net,
                           safety_checker=StableDiffusionSafetyChecker,
                           feature_extractor=CLIPImageProcessor)
    return pipe, generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMAGDressing_v1')

    parser.add_argument('--model_ckpt',
                        default="ckpt/IMAGDressing-v1_512.pt",
                        type=str)
    parser.add_argument('--cloth_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default="./output_sd_inpaint")
    parser.add_argument('--device', type=str, default="cuda:1")
    args = parser.parse_args()

    # svae path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # prepare pipline
    pipe, generator = prepare(args)
    # prepare mask model
    parsing_model = Parsing(1)
    openpose_model = OpenPose(1)
    print('====================== pipe load finish ===================')

    num_samples = 1
    clip_image_processor = CLIPImageProcessor()

    img_transform = transforms.Compose([
        transforms.Resize([640, 512], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    prompt = 'A beautiful model'
    null_prompt = ''
    negative_prompt = 'bare, naked, nude, undressed, monochrome, lowres, bad anatomy, worst quality, low quality'
    
    cloth_files = [f for f in os.listdir(args.cloth_path) if os.path.isfile(os.path.join(args.cloth_path, f))]
    model_files = [f for f in os.listdir(args.model_path) if os.path.isfile(os.path.join(args.model_path, f))]
    for model_file in model_files:
        for cloth_file in cloth_files:
            cloth_path = os.path.join(args.cloth_path, cloth_file)

            clothes_img = Image.open(cloth_path).convert("RGB")
            clothes_img = resize_img(clothes_img)

            vae_clothes = img_transform(clothes_img).unsqueeze(0)
            ref_clip_image = clip_image_processor(images=clothes_img, return_tensors="pt").pixel_values

            model_path = os.path.join(args.model_path, model_file)
            model_image = load_image(model_path)
            keypoints = openpose_model(model_image.resize((384, 512)))
            model_parse, _ = parsing_model(model_image.resize((384, 512)))
            mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)

            mask_image = mask.resize((512, 512))
            model_image = model_image.resize((512, 512))
            control_image = make_inpaint_condition(model_image, mask_image)

            output = pipe(
                ref_image=vae_clothes,
                prompt=prompt,
                ref_clip_image=ref_clip_image,
                null_prompt=null_prompt,
                negative_prompt=negative_prompt,
                image=model_image,
                mask_image=mask_image,
                control_image=control_image,
                width=512,
                height=640,
                num_images_per_prompt=num_samples,
                guidance_scale=5.0,
                image_scale=1.0,
                generator=generator,
                num_inference_steps=50,
            ).images

            save_output = []
            save_output.append(output[0])
            save_output.insert(0, clothes_img.resize((512, 640), Image.BICUBIC))

            grid = image_grid(save_output, 1, 2)
            output_filename = f"{os.path.splitext(model_file)[0]}_{os.path.splitext(cloth_file)[0]}.png"
            grid.save(os.path.join(output_path, output_filename))