import argparse
import logging
import time
import itertools

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
from diffusers import AutoencoderKL,  UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from adapter.resampler import Resampler
from IGPair import VDDataset, collate_fn
from adapter.attention_processor import CacheAttnProcessor2_0,  CAttnProcessor2_0, RefSAttnProcessor2_0

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_text_model_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--pretrained_adapter_model_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default=None,
        help="Path to dataset json file.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        '--clip_penultimate',
        type=bool,
        default=False,
        help='Use penultimate CLIP layer for text embedding'
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0.05, help="noise_offset."
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=0, help="noise_offset."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--num_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


def load_training_checkpoint(model, load_dir, tag=None, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    _, checkpoint_state_dict = model.load_checkpoint(load_dir, tag=tag, **kwargs)
    epoch = checkpoint_state_dict["epoch"]
    last_global_step = checkpoint_state_dict["last_global_step"]
    del checkpoint_state_dict
    return (epoch, last_global_step)


def count_model_params(model):
    return sum([p.numel() for p in model.parameters()]) / 1e6


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


class SDModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, unet, ref_unet, proj, adapter_modules) -> None:
        super().__init__()

        self.unet = unet
        self.ref_unet = ref_unet
        self.proj = proj
        self.adapter_modules = adapter_modules

    def forward(self, encoder_hidden_states, latents, ref_latents, clip_image_embeddings, timesteps):
        ref_timesteps = torch.zeros_like(timesteps)
        cloth_proj_embed = self.proj(clip_image_embeddings)

        _ = self.ref_unet(
            ref_latents,
            ref_timesteps,
            cloth_proj_embed,
            return_dict=False,
        )
        # get cache tensors
        sa_hidden_states = {}
        for name in self.ref_unet.attn_processors.keys():
            sa_hidden_states[name] = self.ref_unet.attn_processors[name].cache["hidden_states"]

        # get noise predictions
        # Predict the noise residual and compute loss
        noise_pred = self.unet(
            latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs={
                "sa_hidden_states": sa_hidden_states,
            }
        ).sample

        return noise_pred


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        log_with=args.report_to,
        project_dir=logging_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_path)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    # load ipa weight
    ipa_weight = torch.load(args.pretrained_adapter_model_path, map_location="cpu")
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
    image_proj.load_state_dict(ipa_weight['image_proj'])

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
            attn_procs[name] = RefSAttnProcessor2_0(name, hidden_size)
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ref.weight": st[layer_name + ".to_k.weight"],
                "to_v_ref.weight": st[layer_name + ".to_v.weight"],
            }
            attn_procs[name].load_state_dict(weights)
        else:
            attn_procs[name] = CAttnProcessor2_0(name, hidden_size=hidden_size,
                                                 cross_attention_dim=cross_attention_dim)  # .to(accelerator.device)]
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    del st

    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    ref_unet.set_attn_processor(
        {name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})  # set cache

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    image_proj.requires_grad_(True)
    ref_unet.requires_grad_(True)

    sd_model = SDModel(unet, ref_unet, image_proj, adapter_modules)


    params_to_opt = itertools.chain(sd_model.proj.parameters(), sd_model.ref_unet.parameters(),
                                    sd_model.adapter_modules.parameters())
    accelerator.print("Trainable parameters: proj:{:.2f}M, ref_unet:{:.2f}M, adapter_modules:{:.2f}M".format(
        count_model_params(sd_model.proj), count_model_params(sd_model.ref_unet),
        count_model_params(sd_model.adapter_modules)))
    # accelerator.print("Trainable parameters: {:.2f}M".format(len(params_to_opt)))
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    if (
            accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        # use deepspeed config
        optimizer = DummyOptim(
            params_to_opt,
            lr=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["lr"],
            weight_decay=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["weight_decay"]
        )

    # TODO (patil-suraj): load scheduler using args
    noise_scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing", prediction_type="epsilon",
    )

    dataset = VDDataset(
        [
            args.dataset_json_path,
        ],
        tokenizer,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, sampler=train_sampler, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=4,
    )

    if accelerator.state.deepspeed_plugin is not None:
        # here we use agrs.gradient_accumulation_steps
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        # use deepspeed scheduler
        lr_scheduler = DummyScheduler(
            optimizer,
            warmup_num_steps=accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"]["params"][
                "warmup_num_steps"]
        )

    if (
            accelerator.state.deepspeed_plugin is not None
            and accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto"
    ):
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size

    sd_model, optimizer, lr_scheduler = accelerator.prepare(sd_model, optimizer, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin is None:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    else:
        if accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]:
            weight_dtype = torch.float16
        elif accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]:
            weight_dtype = torch.bfloat16
    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # New Code #
        # Loads the DeepSpeed checkpoint from the specified path
        last_epoch, last_global_step = load_training_checkpoint(
            sd_model,
            args.resume_from_checkpoint,
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        starting_epoch = last_epoch
        global_steps = last_global_step

    for epoch in range(starting_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        step = 0
        begin = time.perf_counter()
        for batch in train_dataloader:
            load_data_time = time.perf_counter() - begin
            # Convert images to latent space
            with torch.no_grad():
                latents = vae.encode(
                    batch["vae_person"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                ref_latents = vae.encode(
                    batch["vae_clothes"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                ref_latents = ref_latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            if args.noise_offset > 0:
                noise += args.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1),
                    device=latents.device,
                )
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)


            clip_images = []
            for clip_image, drop_image_embed in zip(batch["clip_image"], batch["drop_image_embed"]):
                if drop_image_embed == 1:
                    clip_images.append(torch.zeros_like(clip_image))
                else:
                    clip_images.append(clip_image)
            clip_images = torch.stack(clip_images, dim=0)

            with torch.no_grad():
                # print()
                image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype),
                                             output_hidden_states=True).hidden_states[-2]

            with torch.no_grad():
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

            if noise_scheduler.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(
                    latents, noise, timesteps
                )
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.prediction_type}"
                )

            model_pred = sd_model(encoder_hidden_states, noisy_latents, ref_latents, image_embeds, timesteps)

            if args.snr_gamma == 0:

                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective requires that we add one to SNR values before we divide by them.
                    snr = snr + 1
                mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                )
                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction="none"
                )
                loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                )
                loss = loss.mean()

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item()

            # Backpropagate
            accelerator.backward(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # do nothing
                lr_scheduler.step()  # only for not deepspeed lr_scheduler
                optimizer.zero_grad()  # do nothing

                if accelerator.sync_gradients:
                    accelerator.log({"train_loss": train_loss / args.gradient_accumulation_steps}, step=global_steps)
                    train_loss = 0.0

            if accelerator.is_main_process:
                logging.info(
                    "Epoch {}, step {},  step_loss: {}, lr: {}, time: {}, data_time: {}".format(
                        epoch, global_steps, loss.detach().item(), lr_scheduler.get_lr()[0],
                        time.perf_counter() - begin, load_data_time)
                )
            global_steps += 1
            step += 1

            # checkpoint
            if isinstance(checkpointing_steps, int):
                if global_steps % checkpointing_steps == 0:
                    checkpoint_model(args.output_dir, global_steps, sd_model, epoch, global_steps)

            # stop training
            if global_steps >= args.max_train_steps:
                break
            begin = time.perf_counter()

    accelerator.wait_for_everyone()
    # Save last model
    checkpoint_model(args.output_dir, global_steps, sd_model, epoch, global_steps)

    accelerator.end_training()


if __name__ == "__main__":
    main()
