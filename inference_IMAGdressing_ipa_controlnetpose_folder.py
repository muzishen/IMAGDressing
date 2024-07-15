import diffusers
from dressing_sd.pipelines.IMAGDressing_v1_pipeline_ipa_controlnet import IMAGDressing_v1
import os
import torch

from adapter.attention_processor import CacheAttnProcessor2_0, RefSAttnProcessor2_0, CAttnProcessor2_0

from PIL import Image
from diffusers import ControlNetModel, UNet2DConditionModel, \
    AutoencoderKL, DDIMScheduler
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from adapter.attention_processor import RefSAttnProcessor2_0, LoraRefSAttnProcessor2_0,  IPAttnProcessor2_0, LoRAIPAttnProcessor2_0 
import argparse
from adapter.resampler import Resampler
from insightface.app import FaceAnalysis
import cv2
from insightface.utils import face_align


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
        # lora_rank = hidden_size // 2 # args.lora_rank
        if cross_attention_dim is None:
            attn_procs[name] = LoraRefSAttnProcessor2_0(name, hidden_size)
        else:
            attn_procs[name] = LoRAIPAttnProcessor2_0(hidden_size=hidden_size,
                                                      cross_attention_dim=cross_attention_dim,
                                                      scale=1.0, rank=128,
                                                      num_tokens=4)

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

    control_net_openpose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose",
                                                           torch_dtype=torch.float16).to(device=args.device)
    pipe = IMAGDressing_v1(unet=unet, reference_unet=ref_unet, vae=vae, tokenizer=tokenizer,
                            text_encoder=text_encoder, image_encoder=image_encoder,
                            ip_ckpt=args.ip_ckpt,
                            ImgProj=image_proj, controlnet=control_net_openpose,
                            scheduler=noise_scheduler,
                            safety_checker=StableDiffusionSafetyChecker,
                            feature_extractor=CLIPImageProcessor)
    return pipe, generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMAGDressing_v1')
    parser.add_argument('--ip_ckpt',
                        default="ckpt/ip-adapter-faceid-plusv2_sd15.bin",
                        type=str)
    parser.add_argument('--model_ckpt',
                        default="ckpt/IMAGDressing-v1_512.pt",
                        type=str)
    parser.add_argument('--cloth_path', type=str, required=True)
    parser.add_argument('--face_path', default=None, type=str)
    parser.add_argument('--pose_path', default=None, type=str)
    parser.add_argument('--output_path', type=str, default="./output_sd2")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    # svae path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pipe, generator = prepare(args)
    print('====================== pipe load finish ===================')

    num_samples = 1
    clip_image_processor = CLIPImageProcessor()

    img_transform = transforms.Compose([
        transforms.Resize([640, 512], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))


    prompt = 'A beautiful woman'
    prompt = prompt + ', best quality, high quality'
    null_prompt = ''
    negative_prompt = 'bare, naked, nude, undressed, monochrome, lowres, bad anatomy, worst quality, low quality'

    cloth_files = [f for f in os.listdir(args.cloth_path) if os.path.isfile(os.path.join(args.cloth_path, f))]
    face_files = [f for f in os.listdir(args.face_path) if os.path.isfile(os.path.join(args.face_path, f))]
    # face_files = ['1.jpg']
    
    for face_file in face_files:
        for cloth_file in cloth_files:
            cloth_path = os.path.join(args.cloth_path, cloth_file)
            clothes_img = Image.open(cloth_path).convert("RGB")
            clothes_img = resize_img(clothes_img)
            vae_clothes = img_transform(clothes_img).unsqueeze(0)
            ref_clip_image = clip_image_processor(images=clothes_img, return_tensors="pt").pixel_values

            face_path = os.path.join(args.face_path, face_file)
            if face_path is not None:

                image = cv2.imread(face_path)
                faces = app.get(image)

                faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)
                face_clip_image = clip_image_processor(images=face_image, return_tensors="pt").pixel_values
            else:
                faceid_embeds = None
                face_clip_image = None

            if args.pose_path is not None:
                pose_image = diffusers.utils.load_image(args.pose_path)
            else:
                pose_image = None

            output = pipe(
                ref_image=vae_clothes,
                prompt=prompt,
                ref_clip_image=ref_clip_image,
                pose_image=pose_image,
                face_clip_image=face_clip_image,
                faceid_embeds=faceid_embeds,
                null_prompt=null_prompt,
                negative_prompt=negative_prompt,
                width=512,
                height=640,
                num_images_per_prompt=num_samples,
                guidance_scale=7.0,
                image_scale=0.9,
                ipa_scale=0.9,
                s_lora_scale= 0.2,
                c_lora_scale= 0.2,
                generator=generator,
                num_inference_steps=50,
            ).images

            save_output = []
            save_output.append(output[0])
            save_output.insert(0, clothes_img.resize((512, 640), Image.BICUBIC))

            grid = image_grid(save_output, 1, 2)
            output_filename = f"{os.path.splitext(face_file)[0]}_{os.path.splitext(cloth_file)[0]}.png"
            grid.save(os.path.join(output_path, output_filename))
