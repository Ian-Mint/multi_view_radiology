# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:28:26 2020

@author: Alan
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms
from Model_2 import Encoder, Decoder
from data_loader import get_loader
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from vocabulary import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    
    # GPU #
    print('CUDA? ', torch.cuda.is_available(), ',  Running on GPU: ',torch.cuda.current_device())
    
    # Create directory to save model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        
    # Image Preprocessing
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Training Data Loader
    data_loader = get_loader(args.image_dir, vocab, args.img_report_path, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    # Validation Data Loader
    
    
    # Build Models
    encoder = Encoder().to(device)
    decoder = Decoder(embed_dim = args.embed_size, decoder_dim = args.hidden_size, vocab_size = len(vocab)).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    encoder_optimizer = torch.optim.Adam( params = filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr = args.encoder_lr)
    decoder_optimizer = torch.optim.Adam( params = filter(lambda p: p.requires_grad, decoder.parameters()),
                                          lr = args.decoder_lr)
    
    encoder.train()
    decoder.train()

    total_step = len(data_loader)
    # Train the models
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            images = images.to(device)
            captions = captions.to(device)
            


            # 
            # Training
            #
            encoded_img = encoder(images)
            scores, cap_sorted, decode_len = decoder(encoded_img, captions, lengths)
            
            
            # Ignore <start>
            targets = cap_sorted[:, 1:]


            # Remove <pad>!!!!!!
            scores = pack_padded_sequence(scores, decode_len, batch_first = True)[0]
            targets = pack_padded_sequence(targets, decode_len, batch_first = True)[0]

             
            # optimization
            loss = criterion(scores, targets)
            decoder_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss.backward()
            
            # update weights
            decoder_optimizer.step()
            encoder_optimizer.step()

            #
            #  Validation
            #  bleu-4 = validation(val_loader, encoder, decoder, criterion, args)
            

            # Print log info
            if i % args.train_log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch+1, args.num_epochs, i+1, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                

def validation(val_loader, encoder, decoder, criterion, args):
    """Perform validation on the validation set"""
    encoder.eval()
    decoder.eval()
    
    references = list()
    hypotheses = list()


    total_step = len(val_loader)
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            
            features = encoder(images)
            scores = decoder(features, captions, lengths)
            
            
            scores_copy = scores.clone()
            # Calculate loss
            loss = criterion(scores, captions)
            
            if i % args.validation_log_step == 0:
                print('[Validation] - Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format( i+1, total_step, loss.item(), np.exp(loss.item()))) 
   
    
    print('[Done]\n')
    
            
            # create references
            #for j in range(targets.shape[0]):
            #    img_caps = targets[j].tolist()
            
            
            # create hypotheses
            # _, preds = torch.max(scores_copy, dim=2)
            #preds = preds.tolist()
            #temp_preds = list()
            #for j, p in enumerate(preds):
            #    temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            #preds = temp_preds
            #hypotheses.extend(preds)
            
            
            
        # calculate BLEU-4 scores
        #bleu4_score = corpus_bleu(references, hypotheses)
        #print(BLEU-4 - {bleu}\n'.format(bleu=bleu4))
        
        
    #return bleu4_score
        
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--img_report_path', type=str, default='data/Img_Report.csv', help='path for img name vs findings')
    parser.add_argument('--image_dir', type=str, default='data/png', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/ecgen-radiology', help='path for train annotation json file')
    
    # Log parameters
    parser.add_argument('--train_log_step', type=int , default=1, help='step size for prining training log info')
    parser.add_argument('--validation_log_step', type=int , default=5, help='step size for prining validation log info')
    parser.add_argument('--save_step', type=int , default=10000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in LSTM')
    
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--encoder_lr', type=float, default=0.0001)
    parser.add_argument('--decoder_lr', type=float, default=0.0004)
    args = parser.parse_args()
    main(args)
