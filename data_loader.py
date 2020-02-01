# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:22:15 2020

@author: Alan
"""

from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset
import torch
import nltk


class OpenI(Dataset):
    def __init__(self, root, vocab, img_name_report, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            vocab: vocabulary wrapper.
            img_name_report: dataframe of image name vs findings
            transform: image transformer.
        """

        self.root = root
        self.vocab = vocab
        self.img_name_report = img_name_report
        self.transform = transform

    def __getitem__(self, idx):
        """Returns one data pair (image and caption)."""

        path = self.img_name_report.iloc[idx]['index']
        caption = self.img_name_report.iloc[idx]['Findings']
        vocab = self.vocab

        image = Image.open(os.path.join(self.root, path + '.png')).convert('RGB')
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
        return len(self.vocab)


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

    return images, targets, lengths


def get_loader(root, vocab, img_report_path, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for Open-i dataset."""
    
    img_findings = pd.read_csv(img_report_path)
    open_i = OpenI(root=root,
                   vocab=vocab,
                   img_name_report=img_findings,
                   transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=open_i,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
