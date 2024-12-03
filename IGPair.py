import json
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor

from random import choice


class VDDataset(Dataset):
    def __init__(
            self,
            json_file,
            tokenizer,
            size=512,
            image_root_path="",
    ):

        if isinstance(json_file, str):
            with open(json_file, 'r') as file:
                self.data = json.load(file)

        elif isinstance(json_file, list):
            for file_path in json_file:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if not hasattr(self, 'data'):
                        self.data = data
                    else:
                        self.data.extend(data)
        else:
            raise ValueError("Input should be either a JSON file path (string) or a list")

        print('=========', len(self.data))

        self.tokenizer = tokenizer
        self.size = size
        self.image_root_path = image_root_path

        self.transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop([640, 512]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        item = self.data[idx]

        person_path = item["image_file"]
        person_img = Image.open(person_path).convert("RGB")
        cloth_path = item["cloth_file"]
        clothes_img = Image.open(cloth_path).convert("RGB")

        text = choice(item['text'])

        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < 0.05:
            drop_image_embed = 1
        elif rand_num < 0.1:  # 0.55: #0.1:
            text = ""
        elif rand_num < 0.15:  # 0.6: #0.15:
            text = ""
            drop_image_embed = 1

        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        null_text_input_ids = self.tokenizer(
            "",
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        vae_person = self.transform(person_img)
        vae_clothes = self.transform(clothes_img)

        clip_image = self.clip_image_processor(images=clothes_img, return_tensors="pt").pixel_values

        return {
            "vae_person": vae_person,
            "vae_clothes": vae_clothes,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "text": text,
            "text_input_ids": text_input_ids,
            "null_text_input_ids": null_text_input_ids,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    vae_person = torch.stack([example["vae_person"] for example in data]).to(
        memory_format=torch.contiguous_format).float()
    vae_clothes = torch.stack([example["vae_clothes"] for example in data]).to(
        memory_format=torch.contiguous_format).float()

    clip_image = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embed = [example["drop_image_embed"] for example in data]

    text = [example["text"] for example in data]
    input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    null_input_ids = torch.cat([example["null_text_input_ids"] for example in data], dim=0)

    return {
        "vae_person": vae_person,
        "vae_clothes": vae_clothes,
        "clip_image": clip_image,
        "drop_image_embed": drop_image_embed,
        "text": text,
        "input_ids": input_ids,
        "null_input_ids": null_input_ids,
    }
