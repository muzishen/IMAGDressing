# Metric CAMI Calculation

## 1. CAMI-U calculation
```sh
python eval.py --cloth_path 'input cloth path' --cloth_mask_path 'generate cloth path'
```

## 1. CAMI-S calculation
```sh
python eval_s.py --cloth_path 'input cloth path' --cloth_mask_path 'generate cloth path' --model_path 'generate model path' --pose_path 'input pose path' --face_path 'input face path' --text_prompts 'input text prompts list'
```
