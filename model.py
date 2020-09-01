# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:33:59 2020

@author: Alan
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        '''
        Load the pretrain
        '''
        super(Encoder, self).__init__()
        resnet = models.resnet18(pretrained=True)      
        
        self.encoded_image_size = encoded_image_size
        self.resnet = nn.Sequential(*(list(resnet.children())[:-2] )) 
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size,encoded_image_size))
        self.fine_tune()
        
    def forward(self,images):
        '''
        Extract image feature vectors
        '''
        features = self.resnet(images)    
        features = self.adaptive_pool(features)
        features = features.permute(0,2,3,1) # (batch_size, encoded_image_size, encoded_image_size, 512) 
        return features

    def fine_tune(self, fine_tune=True):
        '''
        Allow or avoid the computation of gradient for convolutional blocks
        '''
        for p in self.resnet.parameters():
            p.requires_grad = False

        # block 2 through 4 
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = True

class Decoder(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim = 512, dropout=0.5):
        super(Decoder,self).__init__()
        
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim) # embedding layer
        self.dropout = nn.Dropout(p = dropout)
        
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        
        self.init_weights()

    def init_weights(self):
        '''
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        '''
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)


    def init_hidden_state(self, encoder_out):
        '''
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
                            
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        '''
        mean_encoder_out = encoder_out.mean(dim=1)
        
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        
        return h, c


    def forward(self, encoder_out, encoder_captions, caption_len):
        '''
        Forward Propagation
        '''

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # flatten encoded image
        encoder_out = encoder_out.view(batch_size,-1, encoder_dim) #(batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)



        # Embedding
        embeddings = self.embedding(encoder_captions)  #(batch_size, max_caption_length, embed_dim)

        # Initialize LSTM cell
        h, c = self.init_hidden_state(encoder_out) 

        # Ignore <end>
        decode_lengths = (caption_len-1).tolist()

        
        # Create tensors to hold word predictions scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        #alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)


        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            #attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],h[:batch_size_t])        
            
            #gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                                                
            #attention_weighted_encoding = gate * attention_weighted_encoding
                                                                                        
            #h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
            #                        (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
         
            #print(embeddings[:batch_size_t, t,:].shape)
            #print(encoder_out[:batch_size_t].sum(dim=1).shape)


            h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], encoder_out[:batch_size_t].sum(dim=1)],dim =1),
                                    (h[:batch_size_t], c[:batch_size_t]))
            
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            
            predictions[:batch_size_t, t, :] = preds
            
            #alphas[:batch_size_t, t, :] = alpha


        #return predictions, encoded_captions, decode_lengths, alphas, sort_ind
        return predictions, encoder_captions, decode_lengths


