import json
import os
from diffusers import DiffusionPipeline
import torch
import warnings
warnings.filterwarnings('ignore')
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mod', type=int)
    parser.add_argument('--div', type=int, default=8)
    parser.add_argument('--num', type=int, default=40)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    num = args.num
    mod = args.mod
    div = args.div

    finished = [int(f.split()[1]) for f in open('finished_8-1.out').read().split('\n')[:-1]]

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
        if idx % div != mod:
            continue
        if idx in finished and not args.overwrite:
            continue
        id = caption['id']
        if os.path.exists(os.path.join(dst_dir, f'{id:010d}_{num-1:02d}.png')):
            continue
        imgs = pipeline([caption['caption'] for _ in range(num)])[0]
        for i in range(num):
            imgs[i].save(
                os.path.join(dst_dir, f'{id:010d}_{i:02d}.png')
            )
        print(mod, idx, len(captions), flush=True)
