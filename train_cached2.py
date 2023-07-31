import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import nltk
import evaluate
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from hf_data import CachedCOCO
import json
from models import CachedFeatureDecoderModel, DINOPretrained, BaseConfig, get_activation
from mmengine.config import Config
from mmdet.apis import init_detector
import argparse


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
        if args.mode == 'backbone':
            model_inputs['features'].append(obj[0]['backbone'].reshape(1, 1536, 25*25).permute(0, 2, 1))
        else:
            model_inputs['features'].append(obj[0]['features[5]'])
        model_inputs['labels'].append(obj[1])
    model_inputs['labels'] = tokenization_fn(model_inputs['labels'])
    model_inputs['features'] = torch.cat(model_inputs['features']).cpu()
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
    parser.add_argument('featdir', type=str)
    parser.add_argument('hidden_size', type=int)
    parser.add_argument('mode', type=str)
    parser.add_argument('--max_per_img', type=int, default=50)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()
    max_per_img = args.max_per_img
    expname = args.expname
    logdir = os.path.join(args.logdir, expname)
    if os.path.exists("/project/lt200060-capgen/coco"):
        feature_dir = f'/project/lt200060-capgen/palm/hf-captioning/features/{args.featdir}'
        vit_model = "/project/lt200060-capgen/palm/huggingface/vit-base-patch16-224-in21k"
        text_decode_model = "/project/lt200060-capgen/palm/huggingface/gpt2"
        src_dir = "/project/lt200060-capgen/coco/images"
        train_json = '/project/lt200060-capgen/coco/annotations/captions_train2017.json'
        val_json = '/project/lt200060-capgen/coco/annotations/captions_val2017.json'
        output_dir = os.path.join('/project/lt200060-capgen/palm/hf-captioning/', expname)
        config_file = '/home/nhongcha/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
        detector_weight = '/project/lt200060-capgen/palm/pretrained/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
        bleu_path = '/home/nhongcha/hf-caption/bleu/bleu.py'
        bs = 16
        workers = 0
    elif os.path.exists("/media/palm/Data/capgen/"):
        feature_dir = f'/project/lt200060-capgen/palm/hf-captioning/features/{args.featdir}'
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        src_dir = "/media/palm/Data/capgen/"
        train_json = '/home/palm/data/coco/annotations/annotations/captions_train2017.json'
        val_json = '/home/palm/data/coco/annotations/annotations/captions_val2017.json'
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
        detector_weight = ''
        output_dir = os.path.join('/tmp/out/mm_dino_8x8')
        bleu_path = '/home/nhongcha/hf-caption/bleu/bleu.py'
        bs = 1
        workers = 0
    else:
        feature_dir = f'/tmp/{args.featdir}'
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        train_json = '/home/palm/data/coco/annotations/annotations/captions_train2017.json'
        val_json = '/home/palm/data/coco/annotations/annotations/captions_val2017.json'
        src_dir = "/home/palm/data/coco/images"
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
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

    model = CachedFeatureDecoderModel(
        None,
        DINOPretrained(BaseConfig(hidden_size=args.hidden_size)),
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

    train_set = CachedCOCO(
        train_json,
        feature_dir,
        digits=9
    )
    print(len(train_set), flush=True)
    valid_set = CachedCOCO(
        val_json,
        feature_dir,
        digits=9
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
