import json
import os
from diffusers import DiffusionPipeline
import torch
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    num = 40
    train_json = '/project/lt200060-capgen/coco/annotations/captions_train2017.json'
    src_dir = "/project/lt200060-capgen/coco/images"
    dst_dir = "/project/lt200060-capgen/palm/sd/images"
    json_file = json.load(open(train_json))
    captions = json_file['annotations']
    
    sd_model = "/project/lt200060-capgen/palm/huggingface/stable-diffusion-v1-5"
    pipeline = DiffusionPipeline.from_pretrained(sd_model, torch_dtype=torch.float16)
    pipeline.to("cuda")

    for idx, caption in enumerate(captions):
        image_id = caption['image_id']
        imgs = pipeline([caption['caption'] for _ in range(num)])[0]
        for i in range(num):
            imgs[i].save(
                os.path.join(dst_dir, f'{image_id}_{i:02d}.png')
            )
        print(idx, len(captions))

