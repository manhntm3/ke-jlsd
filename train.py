"""Train and evaluate the model"""

import argparse
import random
import logging
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
import numpy as np

from transformers import BertTokenizer, BertModel, BertTokenizerFast
from pytorch_pretrained_bert import BertForTokenClassification

from data_loader import DataLoader
from evaluate import evaluate
import utils

def train(model, data_iterator, optimizer, scheduler, params):
    """Train the model on `steps` batches"""
    # set model to training mode
    model.train()

    # a running average object for loss
    loss_avg = utils.RunningAverage()
    
    # Use tqdm for progress bar
    t = trange(params.train_steps)

    for i in t:
        # fetch the next training batch
        batch_data, batch_tags = next(data_iterator)
        batch_masks = batch_data.gt(0)
        # compute model output and loss

        loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)

        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
    scheduler.step()
    

def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch."""
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
        
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # Compute number of batches in one epoch
        params.train_steps = params.train_size // params.batch_size
        params.val_steps = params.val_size // params.batch_size

        # data iterator for training
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)
        # Train for one epoch on training set
        train(model, train_data_iterator, optimizer, scheduler, params)

        # data iterator for evaluation
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=False)
        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)

        # Evaluate for one epoch on training set and validation set
        params.eval_steps = params.train_steps
        train_metrics = evaluate(model, train_data_iterator, params, mark='Train')
        params.eval_steps = params.val_steps
        val_metrics = evaluate(model, val_data_iterator, params, mark='Val')
        
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model_to_save.state_dict(),
                               'optim_dict': optimizer_to_save.state_dict()},
                               is_best=improve_f1>0,
                               checkpoint=model_dir)
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("Best val f1: {:05.2f}".format(best_val_f1))
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/Inspec', help="Directory containing the dataset")
    parser.add_argument('--bert_model_dir', default='pretrain/scibert_scivocab_uncased', help="Directory containing the BERT model in PyTorch")
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
    parser.add_argument('--seed', type=int, default=2024, help="random seed for initialization")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before training")  
    parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")

    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.n_gpu = torch.cuda.device_count()
    params.multi_gpu = args.multi_gpu

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("device: {}, n_gpu: {}, 16-bits training: {}".format(params.device, params.n_gpu, False))

    # Create the input data pipeline
    dataset_name = args.data_dir.split("/")[-1]
    logging.info("Loading the datasets " + dataset_name + "...")
    
    # Initialize the DataLoader
    data_loader = DataLoader(args.data_dir, dataset_name, args.bert_model_dir, params, token_pad_idx=0)
    
    # Load training data and test data
    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('dev') if (dataset_name == "SemEval17") else data_loader.load_data('test')

    # Specify the training and validation dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']
    print("##### Datset size: ")
    print(" Training: ", params.train_size)
    print(" Validation: ", params.val_size)
    print("Tags: ", params.tag2idx, params.idx2tag)
    
    # model = BertModel.from_pretrained("./pretrained/scibert_scivocab_uncased")
    model = BertForTokenClassification.from_pretrained(args.bert_model_dir, num_labels=len(params.tag2idx))
    model.to(params.device)

    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if params.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    
    optimizer = Adam(optimizer_grouped_parameters, lr=params.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, args.model_dir, args.restore_file)

