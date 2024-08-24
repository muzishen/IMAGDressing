export HOST_NUM=1
accelerate launch --gpu_ids 0,1,2,3,4,5,6,7,8,9 --use_deepspeed --num_processes 10 \
  --deepspeed_config_file zero_stage2_config.json \
  train.py \
  --pretrained_model_name_or_path="/path_to/stable-diffusion-v1-5/" \
  --pretrained_vae_model_path="/path_to/sd-vae-ft-mse/" \
  --pretrained_adapter_model_path="/path_to/IP-Adapter/ip-adapter-plus_sd15.bin" \
  --image_encoder_path="/path_to/h94/IP-Adapter/models/image_encoder" \
  --dataset_json_path="/path_to/IGPair.json" \
  --clip_penultimate=False \
  --train_batch_size=5 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1000000 \
  --learning_rate=1e-5 \
  --weight_decay=0.01 \
  --lr_scheduler="constant" --num_warmup_steps=2000 \
  --output_dir="/save_path" \
  --checkpointing_steps=10000