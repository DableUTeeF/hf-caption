import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import nltk
import evaluate
import numpy as np
from PIL import Image
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from hf_data import Flickr8KDataset
from torch.utils.data import default_collate, DataLoader
import json
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right, CrossEntropyLoss, Seq2SeqLMOutput

class CachedFeatureDecoderModel(VisionEncoderDecoderModel):
    def forward(
            self,
            pixel_values=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_encoder,
        )

        encoder_hidden_states = encoder_outputs[0]
        # torch.save(encoder_hidden_states, 'encoder_hidden_states.pth')

        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # else:
        encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def tokenization_fn(captions, max_target_length=128):
    labels = tokenizer(captions,
                       padding="max_length",
                       max_length=max_target_length,
                       return_tensors='pt').input_ids
    return labels


def collate_fn(batch):
    model_inputs = {'labels': [], 'pixel_values': []}
    for obj in batch:
        model_inputs['labels'].append(obj[1])
        model_inputs['pixel_values'].append(obj[0])
    model_inputs['labels'] = tokenization_fn(model_inputs['labels'])
    model_inputs['pixel_values'] = torch.tensor(np.stack(feature_extractor(model_inputs['pixel_values']).pixel_values))
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
    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


if __name__ == '__main__':
    if os.path.exists("/project/lt200060-capgen/palm/huggingface/vit-base-patch16-224-in21k"):
        image_encoder_model = "/project/lt200060-capgen/palm/huggingface/vit-base-patch16-224-in21k"  # "google/vit-base-patch16-224-in21k"
        text_decode_model = "/project/lt200060-capgen/palm/huggingface/gpt2"
        src_dir = "/project/lt200060-capgen/palm/"
        log_output_dir = "/project/lt200060-capgen/palm/hf-captioning/ori"
        bs = 72
    elif os.path.exists("/media/palm/Data/capgen/"):
        image_encoder_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        src_dir = "/media/palm/Data/capgen/"
        log_output_dir = "/media/palm/Data/capgen/out"
        bs = 1
    else:
        image_encoder_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        src_dir = "/home/palm/data/"
        log_output_dir = "out"
        bs = 8
    metric = evaluate.load("rouge")
    ignore_pad_token_for_loss = True
    config_path = "config.json"
    with open(config_path, "r", encoding="utf8") as f:
        config = json.load(f)

    model = CachedFeatureDecoderModel.from_encoder_decoder_pretrained(image_encoder_model, text_decode_model)
    feature_extractor = ViTImageProcessor.from_pretrained(image_encoder_model)
    tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
    # GPT2 only has bos/eos tokens but not decoder_start/pad tokens
    tokenizer.pad_token = tokenizer.eos_token

    # update the model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    output_dir = os.path.join(log_output_dir, "vit-gpt-model")
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    train_hyperparams = {
        "batch_size": config["batch_size"]["train"],
        "shuffle": True,
        "num_workers": 1,
        "drop_last": True
    }
    valid_hyperparams = {
        "batch_size": config["batch_size"]["eval"],
        "shuffle": False,
        "num_workers": 1,
        "drop_last": True
    }

    train_set = Flickr8KDataset(config, src_dir, training=True)
    print(len(train_set), flush=True)
    valid_set = Flickr8KDataset(config, src_dir, training=False)
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
        dataloader_num_workers=1,
        logging_strategy='steps',
        logging_steps=100,
        disable_tqdm=True,
        report_to=['tensorboard']
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collate_fn,
    )
    trainer.train()
