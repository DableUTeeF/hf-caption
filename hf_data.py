import os
import json

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


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

    def _create_input_label_mappings(self, data):
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
