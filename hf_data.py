import os
import json

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class Flickr8KDataset(Dataset):
    """"Represents dataloader for the Flickr8k dataset.

    Data is stored in following format:
        image_name: associated caption
    Each image has maximum 5 different captions.
    """

    def __init__(self, config, src_dir, training=True):
        """Initializes the module.
        
        Arguments:
            config (object): Contains dataset configuration
            path (str): Location where image captions are stored
        """
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
        """Creates (image, description) pairs.

        Arguments:
            data (list of str): Each element consists out of image file name and appropriate caption
                Elements are organized in the following format: 'image_name[SPACE]caption'
        Returns:
            processed_data (list of tuples): Each tuple is organized in following format: (image_name, caption)
        """
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
        return Image.open(os.path.join(self.src_dir, self._image_specs, image)), caption
