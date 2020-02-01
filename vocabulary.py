# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:49:03 2020

@author: Alan
"""


import nltk
nltk.download('punkt')
from collections import Counter
import pandas as pd
import argparse
import pickle

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
    
def build_vocab(imgfindings, threshold = 0):
    
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
    
def main(args):

    imgfindings = pd.read_csv(args.img_cap_path)
    vocab = build_vocab(imgfindings)
    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(args.vocab_path))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_cap_path', type=str, 
                        default='data/Img_Report.csv', 
                        help='path for image name and annotation file')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=0, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
    
    
    
    
