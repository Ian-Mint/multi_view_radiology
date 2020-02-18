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
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from vocabulary import Vocabulary
import pandas as pd
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    # GPU 
    if torch.cuda.is_available():
        print('Using GPU')
    else:
        print('Using CPU')
    
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

    # Split data into 'Train', 'Validate'

    img_name_report = pd.read_csv(args.img_report_path)
    # TODO: parameterize or remove for real training
    img_indices = np.random.choice(img_name_report.index, size=10, replace=False)
    img_name_report = img_name_report.loc[img_indices]

    data_total_size = len(img_name_report)
    print('Data Total Size:{}'.format(data_total_size))
    train_size = int(data_total_size * 0.8)
    train_data = img_name_report.sample(n=train_size)
    img_name_report.drop(list(train_data.index), inplace=True)
    val_data = img_name_report
    train_data.reset_index(level=0, inplace=True)
    val_data.reset_index(level=0, inplace=True)
    print('Training Data:{}'.format(len(train_data)))
    print('Valdiation Data:{}'.format(len(val_data)))

    # Data Loader
    train_loader = get_loader(args.image_dir, vocab, train_data, 
                             transform, args.train_batch_size,
                             shuffle=True, num_workers=args.num_workers, split='Train') 
    val_loader = get_loader(args.image_dir, vocab, val_data,
                            transform, args.val_batch_size,
                            shuffle=True, num_workers=args.num_workers, split='Val')

    
    # Build Models
    encoder = Encoder().to(device)
    decoder = Decoder(embed_dim = args.embed_size, decoder_dim = args.hidden_size, vocab_size = len(vocab)).to(device)
    # Tensorboard Initialize
    if args.tensorboard:
        writer = SummaryWriter(os.path.join(args.log_dir, ts))
        writer.add_text("args", str(args))
        
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    encoder_optimizer = torch.optim.Adam( params = filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr = args.encoder_lr)
    decoder_optimizer = torch.optim.Adam( params = filter(lambda p: p.requires_grad, decoder.parameters()),
                                          lr = args.decoder_lr)
    
    encoder.train()
    decoder.train()

    total_step = len(train_loader)
    
    
    # Train the models
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(train_loader):
            
            images = images.to(device)
            captions = captions.to(device)

            # Training
            encoded_img = encoder(images)
            scores, cap_sorted, decode_len = decoder(encoded_img, captions, lengths)
            
            
            # Ignore <start>
            targets = cap_sorted[:, 1:]

            # Remove <pad>
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

            
            # Bleu on training set
            if i % args.train_bleu_step == 0:
                print('--evaluate bleu score on training set--')
                references = list()
                hypotheses = list()
                with torch.no_grad():   
                    for ii, (img, caps, lens) in enumerate(train_loader):
                         img = img.to(device)
                         caps = caps.to(device)
                         enc_img = encoder(img)
                         sc, cap_so, dec_len = decoder(enc_img, caps, lens)

                         ta = cap_so[:,1:]
                         sc_copy = sc.clone()
                         sc = pack_padded_sequence(sc, dec_len, batch_first=True)[0]
                         ta = pack_padded_sequence(ta, dec_len, batch_first=True)[0]
                         
                         # References
                         for j in range(cap_so.shape[0]):
                             img_cap = cap_so[j].tolist()
                             img_captions = [w for w in img_cap if w not in {vocab('<start>'), vocab('<end>')}]
                             references.append([img_captions])

                         # Hypotheses
                         _, preds = torch.max(sc_copy, dim=2)
                        

                         preds = preds.tolist()
                         temp_preds = list()
                         for j, p in enumerate(preds):
                             temp_preds.append(preds[j][:dec_len[j]])
                        
                         preds = temp_preds
                         hypotheses.extend(preds)

                         assert len(references) == len(hypotheses)

                    #calculate average BLEU-4 scores
                    bleu_score_train = 0
                    for ii in range(len(references)):
                         chencherry = SmoothingFunction()
                         current_score = sentence_bleu(references[ii], hypotheses[ii],smoothing_function=chencherry.method1)
                         bleu_score_train += current_score
                   
                    bleu4_score_train = bleu_score_train/len(references)
                    print('[Train] BLEU-4 - {bleu}\n'.format(bleu=bleu4_score_train))
                 
                    
                    # Tensorboard
                    if args.tensorboard:
                        writer.add_scalar("Train Bleu", bleu4_score_train, epoch*total_step + i)
            
            # Validation  
            if i % args.validation_step == 0:
                bleu_score, loss_val = validation(val_loader, encoder, decoder, criterion, vocab)
                
                if args.tensorboard:
                    writer.add_scalar("Val Bleu", bleu_score, epoch*total_step + i)
                    writer.add_scalar("Val Loss", loss_val, epoch*total_step + i)

            # Print log info
            if i % args.train_log_step == 0:
                print('[Training] -  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch+1, args.num_epochs, i+1, total_step, loss.item(), np.exp(loss.item()))) 
            
            # Tensorboard
            if args.tensorboard:
                writer.add_scalar("Train Loss", loss.item(), epoch*total_step + i)
          

            # Save the model checkpoints
            #if (i+1) % args.save_step == 0:
            #    torch.save(encoder.state_dict(), os.path.join(
            #        args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            #    torch.save(decoder.state_dict(), os.path.join(
            #        args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                

def validation(val_loader, encoder, decoder, criterion, vocab):
    """Perform validation on the validation set"""
     
    encoder.eval()  # no Dropout and BatchNorm
    decoder.eval()
    
    references = list()  # true captions
    hypotheses = list()  # predictions

    total_step = len(val_loader)
    loss_tmp = 0
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)            
            
            encoded_img = encoder(images)
            scores, cap_sorted, decode_len = decoder(encoded_img, captions, lengths)


            # Ignore <start>
            targets = cap_sorted[:, 1:]

            # Remove <pad>
            scores_copy = scores.clone()
            
            scores = pack_padded_sequence(scores, decode_len, batch_first = True)[0]
            targets = pack_padded_sequence(targets, decode_len, batch_first = True)[0]

            # Calculate loss
            loss = criterion(scores, targets)
            loss_tmp += loss.item()
            if i % args.validation_log_step == 0:
                print('     [Validation] - Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format( i+1, total_step, loss.item(), np.exp(loss.item()))) 
   

            # References
            for j in range(cap_sorted.shape[0]):
                img_cap = cap_sorted[j].tolist() 
                img_captions = [w for w in img_cap if w not in {vocab('<start>'), vocab('<end>')}]
                references.append([img_captions])

           
           # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_len[j]])

            preds = temp_preds
            
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)
        
        #calculate average validation loss
        loss_tmp /= total_step



        #calculate average BLEU-4 scores
        bleu_score = 0
        for i in range(len(references)):
            chencherry = SmoothingFunction()
            current_score = sentence_bleu(references[i], hypotheses[i],smoothing_function=chencherry.method1)
            bleu_score += current_score
    
        bleu4_score = bleu_score/len(references)
        print('     [Validation] BLEU-4 - {bleu}\n'.format(bleu=bleu4_score))
        
        return bleu4_score, loss_tmp
        

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
    parser.add_argument('--train_bleu_step',type=int , default=20, help='step size to evaluate bleu score on training set')
    parser.add_argument('--train_log_step', type=int , default=5, help='step size for prining training log info')
    parser.add_argument('--validation_step', type=int, default=10, help='step size to do validation')
    parser.add_argument('--validation_log_step', type=int , default=10, help='step size for prining validation log info')
    parser.add_argument('--save_step', type=int , default=10000, help='step size for saving trained models')
    parser.add_argument('--tensorboard', type=bool, default=True, help='visualize using tensorboard')
    parser.add_argument('--log_dir',type=str, default='logs',help='path for saving logs')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in LSTM')
    
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--encoder_lr', type=float, default=0.0001)
    parser.add_argument('--decoder_lr', type=float, default=0.0004)
    args = parser.parse_args()
    main(args)
