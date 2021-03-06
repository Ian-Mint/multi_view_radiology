# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:22:15 2020

@author: Alan
"""

import pandas as pd
import os
from torch.utils.data import Dataset
import torch
import nltk

import config

# Choose png or DICOM importer
if config.img_extension == '.dcm':
    import dicom_image as Image
else:
    from PIL import Image


class OpenI(Dataset):
    def __init__(self, root, vocab, img_name_report, split, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            vocab: vocabulary wrapper.
            img_name_report: dataframe of image name vs findings
            transform: image transformer.
        """

        self.root = root
        self.vocab = vocab
        self.data = img_name_report
        self.transform = transform
        self.split = split

    def __getitem__(self, idx):
        """Returns one data pair (image and caption)."""
        
        caption = self.data.iloc[idx]['Findings']
        image_id = self.data.iloc[idx]['index']
        vocab = self.vocab

        image = Image.open(os.path.join(self.root, image_id + config.img_extension)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.tensor(caption)
        

        return image, target

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption)."""

    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for idx, cap in enumerate(captions):
        cap_end = lengths[idx]
        targets[idx, :cap_end] = cap[:cap_end]

    lengths = torch.tensor(lengths)

    return images, targets, lengths


def get_loader(root, vocab, img_report, transform, batch_size, shuffle, num_workers,split):
    """Returns torch.utils.data.DataLoader for Open-i dataset."""

    open_i = OpenI(root=root,
                   vocab=vocab,
                   img_name_report=img_report,
                   split=split,
                   transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=open_i,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              pin_memory = True)
    return data_loader
