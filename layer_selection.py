from mmengine.config import Config
from mmdet.apis import init_detector, DetInferencer
from models import get_activation, DummyLayer
from hf_data import COCOData
import torch
import os
from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead, MODELS


@MODELS.register_module()
class CustomMask2FormerHead(Mask2FormerHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dummy = DummyLayer()

    def forward(self, x,
                batch_data_samples):
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        batch_size = len(batch_img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feats = []
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_feats.append(query_feat)
        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            query_feats.append(query_feat)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                                               (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        _ = self.dummy(query_feats)
        return cls_pred_list, mask_pred_list


if __name__ == '__main__':
    if os.path.exists("/project/lt200060-capgen/coco"):
        vit_model = "/project/lt200060-capgen/palm/huggingface/vit-base-patch16-224-in21k"
        text_decode_model = "/project/lt200060-capgen/palm/huggingface/gpt2"
        src_dir = "/project/lt200060-capgen/coco/images"
        train_json = '/project/lt200060-capgen/coco/annotations/captions_train2017.json'
        val_json = '/project/lt200060-capgen/coco/annotations/captions_val2017.json'
        config_file = '/home/nhongcha/mmdetection/configs/glip/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py'
        detector_weight = '/project/lt200060-capgen/palm/pretrained/glip_l_mmdet-abfe026b.pth'
        bleu_path = '/home/nhongcha/hf-caption/bleu/bleu.py'
    elif os.path.exists("/media/palm/Data/capgen/"):
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        src_dir = "/media/palm/data/coco/images"
        train_json = '/media/palm/data/coco/annotations/captions_train2017.json'
        val_json = '/media/palm/data/coco/annotations/captions_val2017.json'
        config_file = '/home/palm/PycharmProjects/hf-caption/mmdetection/configs/glip/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py'
        detector_weight = '/media/palm/BiggerData/mmdetection/cp/glip_l_mmdet-abfe026b.pth'
        bleu_path = 'bleu'
    else:
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        train_json = '/home/palm/data/coco/annotations/annotations/captions_train2017.json'
        val_json = '/home/palm/data/coco/annotations/annotations/captions_val2017.json'
        src_dir = "/home/palm/data/coco/images"
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic.py'
        detector_weight = '/home/palm/PycharmProjects/mmdetection/cp/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic_20220326_224553-3ec9e0ae.pth'
        bleu_path = 'bleu'
    config = Config.fromfile(config_file)
    config.model.panoptic_head.type = 'CustomMask2FormerHead'
    mmmodel = init_detector(
        config,
        detector_weight,
        device='cpu'
    )
    mmmodel.hooks = {}
    mmmodel.panoptic_head.dummy.register_forward_hook(get_activation(f'query', mmmodel.hooks))

    train_set = COCOData(
        train_json,
        os.path.join(src_dir, 'train2017'),
        training=False,
        rescale=False,
        forced_crop=True,
        config=config_file
    )
    data, _ = train_set[0]
    with torch.no_grad():
        data = {
            'inputs': [data['inputs']],
            'data_samples': [data['data_samples']]
        }
        mmmodel.test_step(data)

    print()
