from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose
import os
from mmengine.config import Config
from mmdet.apis import init_detector
from models import get_activation
from hf_data import COCOData
import torch


@torch.no_grad()
def run(dataset):
    for idx, (data, _) in enumerate(dataset):
        caption = dataset.captions[idx]
        if f'{caption["image_id"]:09d}.pth' in os.listdir(feature_dir):
            try:
                torch.load(os.path.join(feature_dir, f'{caption["image_id"]:09d}.pth'))
                print('skipped', f'{caption["image_id"]:09d}.pth', flush=True)
                continue
            except:
                pass
        data = {
            'inputs': [data['inputs']],
            'data_samples': [data['data_samples']]
        }
        mmmodel.test_step(data)
        bbl = {'backbone': mmmodel.hooks['backbone'][3], 'features[5]': mmmodel.hooks['features[5]']}
        torch.save(bbl, os.path.join(feature_dir, f'{caption["image_id"]:09d}.pth'))


if __name__ == '__main__':
    if os.path.exists("/project/lt200060-capgen/coco"):
        feature_dir = '/project/lt200060-capgen/palm/hf-captioning/features/dino-swin-1333-test'
        config_file = '/home/nhongcha/mmdetection/configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py'
        detector_weight = '/project/lt200060-capgen/palm/pretrained/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'
        train_json = '/project/lt200060-capgen/coco/annotations/captions_train2017.json'
        val_json = '/project/lt200060-capgen/coco/annotations/captions_val2017.json'
        src_dir = "/project/lt200060-capgen/coco/images"
    else:
        feature_dir = '/tmp'
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py'
        detector_weight = '/home/palm/PycharmProjects/mmdetection/cp/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
        train_json = '/home/palm/data/coco/annotations/annotations/captions_train2017.json'
        val_json = '/home/palm/data/coco/annotations/annotations/captions_val2017.json'
        src_dir = "/home/palm/data/coco/images"

    config = Config.fromfile(config_file)
    mmmodel = init_detector(
        config,
        detector_weight,
        device='cuda'
    )
    mmmodel.hooks = {}
    for i in range(len(mmmodel.bbox_head.reg_branches)):
        mmmodel.bbox_head.reg_branches[i][2].register_forward_hook(get_activation(f'features[{i}]', mmmodel.hooks))
    mmmodel.backbone.register_forward_hook(get_activation(f'backbone', mmmodel.hooks))

    train_set = COCOData(
        train_json,
        os.path.join(src_dir, 'train2017'),
        training=False,
        config=config_file,
        rescale=False,
        forced_crop=True
    )
    valid_set = COCOData(
        val_json,
        os.path.join(src_dir, 'val2017'),
        training=False,
        config=config_file,
        rescale=False,
        forced_crop=True
    )
    run(train_set)
    run(valid_set)
