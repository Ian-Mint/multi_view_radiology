# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:33:59 2020

@author: Alan
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self, output_size):
        '''
        Load the pretrain
        '''
        super(Encoder, self).__init__()
        resnet = models.resnet18(pretrained=True)      
        #model = models.vgg16(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1] )) 
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        
    def forward(self,images):
        '''
        Extract image feature vectors
        '''
        with torch.no_grad():
            features = self.resnet(images)
            
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        
        return features
    
    
class Decoder(nn.Module):
    def __init__(self,vocab_size, embed_size, hidden_size):
        '''
        Build LSTM
        '''
        super(Decode,self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.LSTM = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        
        
    def forward(self, sentences, img_features, lengths):
        '''
        Decode images features and generate descriptions for image
        '''
        embeddingMatrix = self.embed(sentences)
        embeddings = torch.cat((img_features.unsqueeze(1), embeddingMatrix), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs