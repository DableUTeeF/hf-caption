from diffusers import DiffusionPipeline
import os
from PIL import Image
from train_baseline import evaluate, VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer, COCOData, Seq2SeqTrainingArguments, Seq2SeqTrainer, compute_metrics, tokenization_fn
import torch


def feature_extraction_fn(image_paths, labels):
    model_inputs = {}
    if torch.rand > 0.5:
        images = [Image.open(image_file) for image_file in image_paths]
    else:
        imgs = pipeline(labels)
        images = imgs[0]

    encoder_inputs = feature_extractor(images=images, return_tensors="pt")
    return encoder_inputs.pixel_values


def collate_fn(batch):
    model_inputs = {'labels': [], 'pixel_values': []}
    for obj in batch:
        model_inputs['labels'].append(obj[1])
        model_inputs['pixel_values'].append(obj[0]['img_path'])
    model_inputs['pixel_values'] = feature_extraction_fn(model_inputs['pixel_values'], labels=model_inputs['labels'])
    model_inputs['labels'] = tokenization_fn(model_inputs['labels'])
    return model_inputs


if __name__ == '__main__':
    max_per_img = 50

    if os.path.exists("/project/lt200060-capgen/coco"):
        vit_model = "/project/lt200060-capgen/palm/huggingface/vit-base-patch16-224-in21k"
        text_decode_model = "/project/lt200060-capgen/palm/huggingface/gpt2"
        src_dir = "/project/lt200060-capgen/coco/images"
        train_json = '/project/lt200060-capgen/coco/annotations/captions_train2017.json'
        val_json = '/project/lt200060-capgen/coco/annotations/captions_val2017.json'
        log_output_dir = "/project/lt200060-capgen/palm/hf-captioning/dino-pre-bbox"
        config_file = '/home/nhongcha/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
        detector_weight = '/project/lt200060-capgen/palm/pretrained/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
        output_dir = os.path.join('/project/lt200060-capgen/palm/hf-captioning/baseline')
        bs = 16
        workers = 4
    elif os.path.exists("/media/palm/Data/capgen/"):
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        src_dir = "/media/palm/Data/capgen/"
        train_json = '/home/palm/data/coco/annotations/annotations/captions_train2017.json'
        val_json = '/home/palm/data/coco/annotations/annotations/captions_val2017.json'
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
        detector_weight = ''
        log_output_dir = "/media/palm/Data/capgen/out"
        output_dir = os.path.join('/project/lt200060-capgen/palm/hf-captioning/baseline')
        bs = 1
        workers = 0
    else:
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        train_json = '/home/palm/data/coco/annotations/annotations/captions_train2017.json'
        val_json = '/home/palm/data/coco/annotations/annotations/captions_val2017.json'
        src_dir = "/home/palm/data/coco/images"
        log_output_dir = "/tmp/out"
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
        detector_weight = '/home/palm/PycharmProjects/mmdetection/cp/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
        output_dir = os.path.join('/project/lt200060-capgen/palm/hf-captioning/baseline')
        bs = 2
        workers = 0
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    ignore_pad_token_for_loss = True

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(vit_model, text_decode_model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(vit_model)
    tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
    tokenizer.pad_token = tokenizer.eos_token
    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipeline.to("cuda:1")

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
        transform=False,
        # config=config_file
    )
    print(len(train_set), flush=True)
    valid_set = COCOData(
        val_json,
        os.path.join(src_dir, 'val2017'),
        training=False,
        transform=False,
        # config=config_file
    )
    print(len(valid_set), flush=True)
    # train_loader = DataLoader(train_set, **train_hyperparams)
    # valid_loader = DataLoader(valid_set, **valid_hyperparams)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=12,
        output_dir=log_output_dir,
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
        data_collator=collate_fn,
    )
    trainer.train()
