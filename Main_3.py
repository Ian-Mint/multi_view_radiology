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
from Model_2 import Encoder, DecoderWithAttention, DecoderWithAttention_Pair
from data_loader import *
import pickle
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import *
import time
import torch.nn.functional as F
import sys

start_epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in 
best_bleu4 = 0.  # BLEU-4 score right now
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = None 


def main(args):
    
    ts = time.strftime('%Y-%b-%d-%H.%M.%S', time.gmtime())
    
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map
    
    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    
    
    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention_Pair(attention_dim=args.attention_dim,
                                       embed_dim=args.emb_dim,
                                       decoder_dim=args.decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=args.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr)
        encoder_front = Encoder()
        encoder_front.fine_tune(fine_tune_encoder)
        encoder_front_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_front.parameters()),
                                               lr=args.encoder_lr) if fine_tune_encoder else None
        
        encoder_side = Encoder()
        encoder_side.fine_tune(fine_tune_encoder)
        encoder_side_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_side.parameters()),
                                             lr=args.encoder_lr) if fine_tune_encoder else None


    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)
    

    # GPU 
    if torch.cuda.is_available():
        print('Using GPU')
    else:
        print('Using CPU')
   
        
    # Image Preprocessing
    transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))]
    )  
    

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder_front = encoder_front.to(device)
    encoder_side = encoder_side.to(device)
    
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.log_dir, ts))
    writer.add_text("args", str(args))
    

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_loader = torch.utils.data.DataLoader(
        OpenI_pair(args.data_folder, args.data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        OpenI_pair(args.data_folder, args.data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    
     # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(decoder_optimizer, 0.5)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_side_optimizer, 0.5)
                adjust_learning_rate(encoder_front_optimizer, 0.5)

        # One epoch's training
        train(args, writer,
              train_loader=train_loader,
              encoder_s=encoder_side,
              encoder_f=encoder_front,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer_s=encoder_side_optimizer,
              encoder_optimizer_f=encoder_front_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4, loss_val = validate(args, 
                                        val_loader=val_loader,
                                        encoder_s=encoder_side,
                                        encoder_f=encoder_front,
                                        decoder=decoder,
                                        criterion=criterion)
        writer.add_scalar("Val Bleu4", recent_bleu4, epoch)
        writer.add_scalar("Val Loss", loss_val, epoch)
        
        

        recent_real_bleu4 = evaluate(args, 3,'VAL',encoder_s=encoder_side,
                                     encoder_f=encoder_front,decoder=decoder,word_map=word_map)[3]
        
        print("real bleu4 is:", recent_real_bleu4)
        writer.add_scalar("Real Val Bleu4", recent_real_bleu4, epoch)
        

        # Check if there was an improvement
        is_best = recent_real_bleu4 > best_bleu4
        best_bleu4 = max(recent_real_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint2(args.data_name, epoch, epochs_since_improvement, encoder_side, encoder_front, decoder, encoder_side_optimizer,encoder_front_optimizer,decoder_optimizer, recent_real_bleu4, is_best)


def train(args, writer, train_loader, encoder_s, encoder_f, decoder, criterion, encoder_optimizer_s, encoder_optimizer_f, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder_s: encoder model for side image
    :param encoder_f: encoder model for front image
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer_s: optimizer to update encoder's weights (if fine-tuning) for side image
    :param encoder_optimizer_f: optimizer to update encoder's weights (if fine-tuning) for front image
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder_s.train()
    encoder_f.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    final_loss = 0
    # Batches
    for i, (imgs_side, imgs_front, caps, caplens) in enumerate(train_loader):
        

        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs_side = imgs_side.to(device)
        imgs_front = imgs_front.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs_side = encoder_s(imgs_side)
        imgs_front = encoder_f(imgs_front)
        
        
        # decoder_intput become two encoded features map
        scores, caps_sorted, decode_lengths,  alphas_s, alphas_f, sort_ind = decoder(imgs_side, imgs_front, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores.data, targets.data)

        # Add doubly stochastic attention regularization
        loss += args.alpha_c * ((1. - alphas_s.sum(dim=1)) ** 2).mean()
        loss += args.alpha_c * ((1. - alphas_f.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer_s is not None:
            encoder_optimizer_s.zero_grad()
            encoder_optimizer_f.zero_grad()
            
        final_loss = loss
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer_s is not None:
                clip_gradient(encoder_optimizer_s, args.grad_clip)
                clip_gradient(encoder_optimizer_f, args.grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer_s is not None:
            encoder_optimizer_s.step()
            encoder_optimizer_f.step()

        # Keep track of metrics
        top5 = accuracy(scores.data, targets.data, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        
        writer.add_scalar("Train Loss", losses.val , epoch* len(train_loader) + i)

        # Print status
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))



def validate(args, val_loader, encoder_s, encoder_f, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder_s: encoder_s model
    :param encoder_f: encoder_f model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder_s is not None:
        encoder_s.eval()
        encoder_f.eval()
        
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs_side, imgs_front, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs_side = imgs_side.to(device)
            imgs_front = imgs_front.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder_s is not None:
                imgs_side = encoder_s(imgs_side)
                imgs_front = encoder_f(imgs_front)
                
            scores, caps_sorted, decode_lengths, alphas_s, alphas_f, sort_ind = decoder(imgs_side, imgs_front, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores= pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets= pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += args.alpha_c * ((1. - alphas_s.sum(dim=1)) ** 2).mean()
            loss += args.alpha_c * ((1. - alphas_f.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()
           

            if i % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4, losses.val

def evaluate(args, beam_size,split,encoder_s, encoder_f,decoder,word_map):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    vocab_size = len(word_map)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    loader = torch.utils.data.DataLoader(
        OpenI_pair(args.data_folder, args.data_name, split=split, transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True)


    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (imgs_side, imgs_front, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        imgs_side = imgs_side.to(device)   # (1, 3, 256, 256)
        imgs_front = imgs_front.to(device)

        # Encode
        encoder_out_side = encoder_s(imgs_side)  # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_out_front = encoder_f(imgs_front)
        
        
        enc_image_size = encoder_out_side.size(1)
        encoder_dim = encoder_out_side.size(3)

        # Flatten encoding
        encoder_out_side = encoder_out_side.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out_side.size(1)
        
        encoder_out_front = encoder_out_front.view(1, -1, encoder_dim)

        # We'll treat the problem as having a batch size of k
        encoder_out_side = encoder_out_side.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        encoder_out_front = encoder_out_front.expand(k, num_pixels, encoder_dim)
        
        

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h_side, c_side = decoder.init_hidden_state(encoder_out_side)
        h_front, c_front = decoder.init_hidden_state(encoder_out_front)
        

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe_side, _ = decoder.attention_side(encoder_out_side, h_side)  # (s, encoder_dim), (s, num_pixels)
            awe_front, _ = decoder.attention_front(encoder_out_front, h_front)
            
            
            gate_side = decoder.sigmoid(decoder.f_beta_side(h_side))  # gating scalar, (s, encoder_dim)
            awe_side = gate_side * awe_side
            gate_front = decoder.sigmoid(decoder.f_beta_front(h_front))  # gating scalar, (s, encoder_dim)
            awe_front = gate_front * awe_front
            
            
            h_side, c_side = decoder.decode_step_side(torch.cat([embeddings, awe_side], dim=1), (h_side, c_side))  
            h_front, c_front = decoder.decode_step_front(torch.cat([embeddings, awe_front], dim=1), (h_front, c_front))  
            
            # concatenate
            h_concat = torch.cat([h_side,h_front], dim=1)

            scores = decoder.fc(h_concat)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h_side = h_side[prev_word_inds[incomplete_inds]]
            c_side = c_side[prev_word_inds[incomplete_inds]]
            h_front = h_front[prev_word_inds[incomplete_inds]]
            c_front = c_front[prev_word_inds[incomplete_inds]]
            
            
            encoder_out_side = encoder_out_side[prev_word_inds[incomplete_inds]]
            encoder_out_front = encoder_out_front[prev_word_inds[incomplete_inds]]
            
            
            
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        
        
        if not complete_seqs_scores:
            continue
            
            
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)
    weights = (1.0 / 1.0,)
    bleu1 = corpus_bleu(references, hypotheses, weights)
    # Calculate BLEU-4 scores
    weights = (1.0 / 2.0, 1.0 / 2.0,)
    bleu2 = corpus_bleu(references, hypotheses, weights)

    weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,)
    bleu3 = corpus_bleu(references, hypotheses, weights)

    bleu4 = corpus_bleu(references, hypotheses)

    return [bleu1,bleu2,bleu3,bleu4]

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path 
    parser.add_argument('--data_folder', type=str, default='data/medical_data_new_pair', help='folder with data files saved by create_input_files.py')
    parser.add_argument('--data_name', type=str, default='iu-x-ray_1_cap_per_img_2_min_word_freq', help='base name shared by data files')
    
    
    # log 
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--log_dir',type=str, default='logs',help='path for saving logs')
       
    # Model Parameters
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--decoder_dim', type=int, default=512)
    
    # Training Parameters
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--encoder_lr', type=float, default=0.0001)
    parser.add_argument('--decoder_lr', type=float, default=0.0004)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=int, default=5, help = 'clip gradients at an absolute value of')
    parser.add_argument('--alpha_c', type=int, default=1, help = 'regularization parameter for doubly stochastic attention, as in the paper')

    
    args = parser.parse_args()
    main(args)
