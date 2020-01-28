# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:22:15 2020

@author: Alan
"""

from PIL import Image
import pandas as pd
import os
#import torch.utils.data import Dataset
import nltk

class OpenI(Dataset):
    def __init__(self, root, vocab, img_name_report, transform = None):
        
        self.root = root
        self.vocab = vocab
        self.img_name_report = img_name_report
        self.transform = transform
          
        
    def _getitem_(self, idx):
        
        path = self.img_name_report[idx]['index']
        caption = self.img_name_report[idx]['Findings']
        
        image = Image.open(os.path.join(self.root, path + '.png')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
    
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
    
    
    
    def __len__(self):
        
        
        pass
        
def get_loader(root, img_findings, transform, batch_size, shuffle, num_workers):
    
    imgfindings = pd.read_csv(img_findings)
    open_i = OpenI(root = root,
                   img_name_report = imgfindings,
                   transform = transform)
    
    
    data_loader = torch.utils.data.DataLoader(dataset = open_i,
                                              batch_size = batch_size,
                                              shuffle = shuffle,
                                              num_workers = num_workers,
                                              )
    
    
    return data_loader



if __name__ == '__main__':
    path = 'preprocessing/Img_Report.csv'
    imgfindings = pd.read_csv(path)