from transformers import AutoTokenizer, Blip2ForConditionalGeneration
import evaluate
import numpy as np


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
    result = {k: round(v * 100, 4) for k, v in rouge_result.items()}
    bleu_result = bleu.compute(predictions=decoded_preds,
                               references=decoded_labels,
                               use_stemmer=True)
    result.update({k: round(v * 100, 4) for k, v in bleu_result.items()})
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/blip2-opt-2.7b')
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')

