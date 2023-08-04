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
        feats = mmmodel.extract_feat(data.unsqueeze(0).cuda())
        torch.save(feats, os.path.join(feature_dir, f'{caption["image_id"]:09d}.pth'))


if __name__ == '__main__':
    if os.path.exists("/project/lt200060-capgen/coco"):
        feature_dir = '/project/lt200060-capgen/palm/hf-captioning/features/glip_l-swin-800-test'
        config_file = '/home/nhongcha/mmdetection/configs/glip/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py'
        detector_weight = '/project/lt200060-capgen/palm/pretrained/glip_l_mmdet-abfe026b.pth'
        train_json = '/project/lt200060-capgen/coco/annotations/captions_train2017.json'
        val_json = '/project/lt200060-capgen/coco/annotations/captions_val2017.json'
        src_dir = "/project/lt200060-capgen/coco/images"
    elif os.path.exists("/media/palm/Data/capgen/"):
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        src_dir = "/media/palm/Data/capgen/"
        train_json = '/media/palm/data/coco/annotations/captions_train2017.json'
        val_json = '/media/palm/data/coco/annotations/captions_val2017.json'
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
        detector_weight = ''
        log_output_dir = "/media/palm/Data/capgen/out"
        output_dir = os.path.join('tmp/baseline')
        bs = 1
        workers = 0
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

    train_set = COCOData(
        train_json,
        os.path.join(src_dir, 'train2017'),
        training=False,
        rescale=False,
        forced_crop=True
    )
    valid_set = COCOData(
        val_json,
        os.path.join(src_dir, 'val2017'),
        training=False,
        rescale=False,
        forced_crop=True
    )
    run(train_set)
    run(valid_set)
