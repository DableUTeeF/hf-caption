from mmengine.config import Config
from mmdet.apis import init_detector, DetInferencer
from models import get_activation
from hf_data import COCOData
import torch
import os

if __name__ == '__main__':
    train_json = '/media/palm/data/coco/annotations/captions_train2017.json'
    config_file = '/home/palm/PycharmProjects/hf-caption/mmdetection/configs/glip/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py'
    detector_weight = '/media/palm/BiggerData/mmdetection/cp/glip_l_mmdet-abfe026b.pth'
    src_dir = "/media/palm/data/coco/images"
    config = Config.fromfile(config_file)
    train_set = COCOData(
        train_json,
        os.path.join(src_dir, 'train2017'),
        training=False,
        rescale=False,
        forced_crop=True
    )
    mmmodel = init_detector(
        config,
        detector_weight,
        device='cpu'
    )
    data, _ = train_set[0]
    with torch.no_grad():
        feats = mmmodel.extract_feat(data.unsqueeze(0).cpu())

    print()
