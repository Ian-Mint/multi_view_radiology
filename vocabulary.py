# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:49:03 2020

@author: Alan
"""

import os
import nltk
from collections import Counter
import pandas as pd

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0  
        
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx+=1
        
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
        
        
    def __len__(self):
        return len(self.word2idx)
    
def build_vocab(img_report, threshold = 0):
    
    counter = Counter()
    
    # add some special token()
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    
    for _ ,id, captions in imgfindings.values:
        tokens = nltk.tokenize.word_tokenize(captions.lower())
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    for word in words:
        vocab.add_word(word)
    return vocab
    
    
if __name__ == '__main__':
    path = 'preprocessing/Img_Report.csv'
    imgfindings = pd.read_csv(path)
    
    vocab = build_vocab(imgfindings)
    
    print("Total vocabulary size: {}".format(len(vocab)))
