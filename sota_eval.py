import json
import evaluate
import numpy as np
from transformers import pipeline
from hf_data import COCOData
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def compute_metrics(decoded_preds, decoded_labels):
    rouge_result = rouge.compute(predictions=decoded_preds,
                                 references=decoded_labels,
                                 use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in rouge_result.items()}
    bleu_result = bleu.compute(predictions=decoded_preds,
                               references=decoded_labels,
                               use_stemmer=True)
    result.update({k: round(v * 100, 4) for k, v in bleu_result.items()})
    return result


if __name__ == '__main__':
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=0)
    src_dir = "/home/palm/data/coco/images"
    val_json = '/home/palm/data/coco/annotations/annotations/captions_val2017.json'
    data = json.load(open(val_json))
    # predicts = {}
    # for image in tqdm(data['images']):
    #     predict = image_to_text(os.path.join(src_dir, 'val2017', image['file_name']))[0]['generated_text']
    #     predicts[image['id']] = predict
    # json.dump(predicts, open('output/decoded_preds.json', 'w'))
    predicts = json.load(open('output/decoded_preds.json'))
    references = {k: [] for k in predicts.keys()}
    for annotation in tqdm(data['annotations']):
        references[annotation['image_id']].append(annotation['caption'])
    decoded_preds, decoded_labels = list(predicts.values()), list(references.values())
    rouge_result = rouge.compute(predictions=decoded_preds,
                                 references=decoded_labels,
                                 use_stemmer=True)
    bleu_result = bleu.compute(predictions=decoded_preds,
                               references=decoded_labels)
    result = {**rouge_result, **bleu_result}

