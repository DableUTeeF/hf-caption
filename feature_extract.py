from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose
import os
from mmengine.config import Config
from mmdet.apis import init_detector
from models import get_activation
from hf_data import COCOData
import torch
from torch import nn
from mmdet.registry import MODELS
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead


class MidLayer(nn.Module):
    @staticmethod
    def forward(self, x):
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        return x


@MODELS.register_module()
class NewRCNNHead(Shared2FCBBoxHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.midlayer = MidLayer()

    def forward(self, x):
        x = self.midlayer(self, x)
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

@torch.no_grad()
def run(dataset):
    for idx, (data, _) in enumerate(dataset):
        caption = dataset.captions[idx]
        data = {
            'inputs': [data['inputs']],
            'data_samples': [data['data_samples']]
        }
        mmmodel.test_step(data)
        feats = mmmodel.hooks['features']
        torch.save(feats.unsqueeze(0), os.path.join(feature_dir, f'{caption["image_id"]:010d}.pth'))


if __name__ == '__main__':
    if os.path.exists("/project/lt200060-capgen/coco"):
        feature_dir = '/project/lt200060-capgen/palm/hf-captioning/features/rcnn-1333-test'
        config_file = '/home/nhongcha/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
        detector_weight = '/project/lt200060-capgen/palm/pretrained/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        train_json = '/project/lt200060-capgen/coco/annotations/captions_train2017.json'
        val_json = '/project/lt200060-capgen/coco/annotations/captions_val2017.json'
        src_dir = "/project/lt200060-capgen/coco/images"
    else:
        feature_dir = '/tmp'
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
        detector_weight = '/home/palm/PycharmProjects/mmdetection/cp/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        train_json = '/home/palm/data/coco/annotations/annotations/captions_train2017.json'
        val_json = '/home/palm/data/coco/annotations/annotations/captions_val2017.json'
        src_dir = "/home/palm/data/coco/images"

    config = Config.fromfile(config_file)
    config.model.roi_head.bbox_head.type = 'NewRCNNHead'
    mmmodel = init_detector(
        config,
        detector_weight,
        device='cuda'
    )
    mmmodel.hooks = {}
    mmmodel.roi_head.bbox_head.midlayer.register_forward_hook(get_activation('features', mmmodel.hooks))

    train_set = COCOData(
        train_json,
        os.path.join(src_dir, 'train2017'),
        training=False,
        config=config_file,
        rescale=False
    )
    valid_set = COCOData(
        val_json,
        os.path.join(src_dir, 'val2017'),
        training=False,
        config=config_file,
        rescale=False
    )
    run(train_set)
    run(valid_set)
