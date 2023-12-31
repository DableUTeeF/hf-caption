import os
import json

from mmengine.config import Config
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose
from diffusers import DiffusionPipeline
import torch


class CachedCOCO(Dataset):
    def __init__(self, json_file, feat_dir, src_dir='/project/lt200060-capgen/coco/images', training=True, digits=10):
        json_file = json.load(open(json_file))
        self.captions = json_file['annotations']
        self.images = {}
        self.digits = f'0{digits}d'
        for image in json_file['images']:
            self.images[image['id']] = image
        self.feat_dir = feat_dir
        if training:
            self.src_dir = os.path.join(src_dir, 'train2017')
        else:
            self.src_dir = os.path.join(src_dir, 'val2017')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        features5 = torch.load(os.path.join(self.feat_dir, f'{caption["image_id"]:{self.digits}}.pth'), map_location='cpu')
        image = self.images[caption["image_id"]]
        return features5, caption['caption'], os.path.join(self.src_dir, image['file_name'])


class COCOData(Dataset):
    def __init__(self, json_file, src_dir, training=True, transform=True, config=None, rescale=True, forced_crop=False):
        json_file = json.load(open(json_file))
        self.config = config
        self.captions = json_file['annotations']
        self.images = {}
        for image in json_file['images']:
            self.images[image['id']] = image
        self.src_dir = src_dir
        scales = [
            (352, 800),
            (384, 800),
            (416, 800),
            (448, 800),
            (480, 800),
            (512, 800),
            (544, 800),
            (576, 800),
            (608, 800),
        ]
        if training and transform:
            if config is not None:
                config = Config.fromfile(config)
                if rescale:
                    config.train_dataloader.dataset.pipeline[3].transforms[0][0].scales = scales
                    config.train_dataloader.dataset.pipeline[3].transforms[1][2].scales = scales
                config.test_dataloader.dataset.pipeline = config.train_dataloader.dataset.pipeline
                self.transform = Compose(get_test_pipeline_cfg(config))
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(800),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        elif transform:
            if config is not None:
                config = Config.fromfile(config)
                if forced_crop:
                    config.test_dataloader.dataset.pipeline[1].scale = (800, 800)
                    config.test_dataloader.dataset.pipeline[1].keep_ratio = False
                elif rescale:
                    config.test_dataloader.dataset.pipeline[1].scale = (800, 480)
                self.transform = Compose(get_test_pipeline_cfg(config))
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(960),
                    transforms.CenterCrop(800),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = lambda x: x

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        image = self.images[caption['image_id']]
        if self.config is not None:
            data_ = dict(img_path=os.path.join(self.src_dir, image['file_name']), img_id=0)
        else:
            data_ = Image.open(os.path.join(self.src_dir, image['file_name'])).convert('RGB')
        data_ = self.transform(data_)
        return data_, caption['caption']


class Flickr8KDataset(Dataset):
    def __init__(self, config, src_dir, training=True):

        self.src_dir = src_dir
        if training:
            path = config["split_save"]["train"]
        else:
            path = config["split_save"]["validation"]
        with open(os.path.join(src_dir, path), "r") as f:
            self._data = [line.replace("\n", "") for line in f.readlines()]
        self._data = self._create_input_label_mappings(self._data)
        self._image_specs = config["image_specs"]["image_dir"]

    @staticmethod
    def _create_input_label_mappings(data):
        processed_data = []
        for line in data:
            tokens = line.split()
            # Separate image name from the label tokens
            img_name, caption_words = tokens[0].split("#")[0], tokens[1:]
            # Construct (X, Y) pair
            pair = (img_name, ' '.join(caption_words))
            processed_data.append(pair)

        return processed_data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        image, caption = self._data[index]
        return os.path.join(self.src_dir, self._image_specs, image), caption


if __name__ == '__main__':
    s = CachedCOCO('/media/palm/data/coco/annotations/captions_val2017.json', '')
    s[0]
