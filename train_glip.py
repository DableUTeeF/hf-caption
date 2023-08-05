import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import nltk
import evaluate
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from hf_data import Flickr8KDataset, COCOData
import json
from models import CachedFeatureDecoderModel, DINOPretrained, BaseConfig, get_activation
from mmengine.config import Config
from mmdet.apis import init_detector
import argparse

t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved
print(f, flush=True)


def tokenization_fn(captions, max_target_length=128):
    labels = tokenizer(captions,
                       padding="max_length",
                       max_length=max_target_length,
                       return_tensors='pt',
                       truncation=True).input_ids
    return labels


@torch.no_grad()
def mm_collate_fn(batch):
    model_inputs = {
        'labels': [],
        'features': []
    }
    for obj in batch:
        model_inputs['labels'].append(obj[1])
        data = obj[0]
        feats = mmmodel.extract_feat(data.unsqueeze(0).cuda())
        feats = torch.cat([f.reshape(1, 256, -1) for f in feats[:2]], 2).permute(0, 2, 1)
        model_inputs['features'].append(feats)
    model_inputs['labels'] = tokenization_fn(model_inputs['labels'])
    model_inputs['features'] = torch.cat(model_inputs['features']).cpu()
    return model_inputs


def std_collate_fn(batch):
    model_inputs = {'labels': [], 'pixel_values': []}
    for obj in batch:
        model_inputs['labels'].append(obj[1])
        model_inputs['pixel_values'].append(obj[0])
    model_inputs['labels'] = tokenization_fn(model_inputs['labels'])
    model_inputs['pixel_values'] = torch.stack(model_inputs['pixel_values'])
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)
    rouge_result = rouge.compute(predictions=decoded_preds,
                                 references=decoded_labels,
                                 use_stemmer=True)
    bleu_result = bleu.compute(predictions=decoded_preds,
                               references=decoded_labels)
    result = {**rouge_result, **bleu_result}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str)
    parser.add_argument('--max_per_img', type=int, default=50)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()
    max_per_img = args.max_per_img
    expname = args.expname
    logdir = os.path.join(args.logdir, expname)
    if os.path.exists("/project/lt200060-capgen/coco"):
        vit_model = "/project/lt200060-capgen/palm/huggingface/vit-base-patch16-224-in21k"
        text_decode_model = "/project/lt200060-capgen/palm/huggingface/gpt2"
        src_dir = "/project/lt200060-capgen/coco/images"
        train_json = '/project/lt200060-capgen/coco/annotations/captions_train2017.json'
        val_json = '/project/lt200060-capgen/coco/annotations/captions_val2017.json'
        output_dir = os.path.join('/project/lt200060-capgen/palm/hf-captioning/', expname)
        config_file = '/home/nhongcha/mmdetection/configs/glip/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py'
        detector_weight = '/project/lt200060-capgen/palm/pretrained/glip_l_mmdet-abfe026b.pth'
        bleu_path = '/home/nhongcha/hf-caption/bleu/bleu.py'
        bs = args.bs
        workers = 0
    elif os.path.exists("/media/palm/Data/capgen/"):
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        src_dir = "/media/palm/data/coco/images"
        train_json = '/media/palm/data/coco/annotations/captions_train2017.json'
        val_json = '/media/palm/data/coco/annotations/captions_val2017.json'
        config_file = '/home/palm/PycharmProjects/hf-caption/mmdetection/configs/glip/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py'
        detector_weight = '/media/palm/BiggerData/mmdetection/cp/glip_l_mmdet-abfe026b.pth'
        output_dir = os.path.join('/tmp/out/mm_dino_8x8')
        bleu_path = 'bleu'
        bs = 1
        workers = 0
    else:
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        train_json = '/home/palm/data/coco/annotations/annotations/captions_train2017.json'
        val_json = '/home/palm/data/coco/annotations/annotations/captions_val2017.json'
        src_dir = "/home/palm/data/coco/images"
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/glip/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py'
        detector_weight = '/home/palm/PycharmProjects/mmdetection/cp/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
        bs = 2
        output_dir = os.path.join('/tmp/out/mm_dino_8x8')
        bleu_path = 'bleu'
        workers = 0
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=args.overwrite)
    os.makedirs(logdir, exist_ok=args.overwrite)
    rouge = evaluate.load("rouge")
    bleu = evaluate.load(bleu_path)
    ignore_pad_token_for_loss = True

    config = Config.fromfile(config_file)
    mmmodel = init_detector(
        config,
        detector_weight,
        device='cuda'
    )

    model = CachedFeatureDecoderModel(
        None,
        DINOPretrained(BaseConfig(hidden_size=256)),
        AutoModelForCausalLM.from_pretrained(text_decode_model)
    )
    # feature_extractor = ViTImageProcessor.from_pretrained(vit_model)
    tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
    tokenizer.pad_token = tokenizer.eos_token

    # update the model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.save_pretrained(output_dir)
    # feature_extractor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    train_set = COCOData(
        train_json,
        os.path.join(src_dir, 'train2017'),
        training=True,
    )
    print(len(train_set), flush=True)
    valid_set = COCOData(
        val_json,
        os.path.join(src_dir, 'val2017'),
        training=False,
    )
    print(len(valid_set), flush=True)
    # train_loader = DataLoader(train_set, **train_hyperparams)
    # valid_loader = DataLoader(valid_set, **valid_hyperparams)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=12,
        output_dir=os.path.join(output_dir, 'train'),
        logging_dir=logdir,
        dataloader_num_workers=workers,
        logging_strategy='steps',
        logging_steps=100,
        disable_tqdm=True,
        # report_to=['tensorboard']
    )
    trainer = Seq2SeqTrainer(
        model=model,
        # tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=mm_collate_fn,
    )
    trainer.train()
