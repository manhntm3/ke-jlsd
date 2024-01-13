"""Data loader"""

import random
import numpy as np
import os
import sys

import torch

from pytorch_pretrained_bert import BertTokenizer
from transformers import AutoTokenizer

import utils

class DataLoader(object):
    def __init__(self, data_dir, data_name, bert_model_dir, params, token_pad_idx=0):
        assert data_name in ['Inspec_two', 'Inspec','SemEval17','task1','task2'] , f"Dataset {data_name} is not supported!"
        self.data_name = data_name
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.token_pad_idx = 0

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag
        self.tag_pad_idx = self.tag2idx['O']

        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)
        # self.tokenizer = BertTokenizerFast.from_pretrained("./pretrained/scibert_scivocab_cased", do_lower_case=False)

    def load_tags(self):
        tags = []
        file_path = os.path.join(self.data_dir, 'tags.txt')
        with open(file_path, 'r') as file:
            for tag in file:
                tags.append(tag.strip())
        return tags

    def load_sentences_tags(self, word_file, tag_file, d):
        """Loads sentences and tags from their corresponding files. 
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sentences = []
        tags = []
        sentence = []
        tag = []
        if 'task' in self.data_name: 
            with open(word_file, 'r') as file:
                for line in file:
                    # replace each token by its index
                    tokens = line.split()
                    sentences.append(self.tokenizer.convert_tokens_to_ids(tokens))
            
            with open(tag_file, 'r') as file:
                for line in file:
                    # replace each tag by its index
                    tag_seq = [self.tag2idx.get(tag) for tag in line.strip().split(' ')]
                    tags.append(tag_seq)
        else:
            with open(word_file, 'r') as file:
                for line in file:
                    if line != "\n":
                        sentence.append(line.split()[0])
                        if line.split()[1][0] == 'B':
                            tag.append('I')
                        else: 
                            tag.append(line.split()[1][0])
                    else:
                        sentences.append(self.tokenizer.convert_tokens_to_ids(sentence))
                        tags.append([self.tag2idx.get(t) for t in tag])
                        sentence = []
                        tag = []
        
        # checks to ensure there is a tag for each token
        assert len(sentences) == len(tags)
        for i in range(len(sentences)):
#             print(sentences[i], tags[i])
            assert len(tags[i]) == len(sentences[i])

        # storing sentences and tags in dict d
        d['data'] = sentences
        d['tags'] = tags
        d['size'] = len(sentences)

    def load_data(self, data_type):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        data = {}
        if 'task' in self.data_name:
            if data_type in ['train', 'val', 'test']:
                sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
                tags_path = os.path.join(self.data_dir, data_type, 'tags.txt')
                self.load_sentences_tags(sentences_file, tags_path, data)
            else:
                raise ValueError("data type not in ['train', 'val', 'test']")
        else:
            word_file = os.path.join(self.data_dir, data_type + ".txt")
            self.load_sentences_tags(word_file,None,data)
        return data

    def data_iterator(self, data, shuffle=False):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'tags' and 'size'
            shuffle: (bool) whether the data should be shuffled
            
        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_tags: (tensor) shape: (batch_size, max_len)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        # one pass over data
        for i in range(data['size']//self.batch_size):
            # fetch sentences and tags
            sentences = [data['data'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
            tags = [data['tags'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in sentences])
            max_len = min(batch_max_len, self.max_len)

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, max_len))
            batch_tags = self.tag_pad_idx * np.ones((batch_len, max_len))

            # copy the data to the numpy array
            for j in range(batch_len):
                cur_len = len(sentences[j])
                if cur_len <= max_len:
                    batch_data[j][:cur_len] = sentences[j]
                    batch_tags[j][:cur_len] = tags[j]
                else:
                    batch_data[j] = sentences[j][:max_len]
                    batch_tags[j] = tags[j][:max_len]

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_tags = torch.tensor(batch_tags, dtype=torch.long)

            # shift tensors to GPU if available
            batch_data, batch_tags = batch_data.to(self.device), batch_tags.to(self.device)
    
            yield batch_data, batch_tags

class UnLabelledDataLoader(object):
    def __init__(self, data_dir, data_name, bert_model_dir, params):
        assert data_name in ['kp20k'] , f"Dataset {data_name} is not supported!"
        self.data_name = data_name
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.token_pad_idx = 0

        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)
        # self.tokenizer = BertTokenizerFast.from_pretrained("./pretrained/scibert_scivocab_cased", do_lower_case=False)

    def load_sentences_tags(self, word_file, d):
        sentences = []
        with open(word_file, 'r') as file:
            for line in file:
                    sentences.append(self.tokenizer.convert_tokens_to_ids(line.split()))

        # storing sentences and tags in dict d
        d['data'] = sentences
        d['size'] = len(sentences)

    def load_data(self, data_type):
        data = {}
        if 'kp20k' in self.data_name:
            word_file = os.path.join(self.data_dir, data_type + ".txt")
            self.load_sentences_tags(word_file, data)

        return data

    def data_iterator(self, data, shuffle=False):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'tags' and 'size'
            shuffle: (bool) whether the data should be shuffled
            
        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_tags: (tensor) shape: (batch_size, max_len)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        # one pass over data
        for i in range(data['size']//self.batch_size):
            # fetch sentences
            sentences = [data['data'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in sentences])
            max_len = min(batch_max_len, self.max_len)

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, max_len))

            # copy the data to the numpy array
            for j in range(batch_len):
                cur_len = len(sentences[j])
                if cur_len <= max_len:
                    batch_data[j][:cur_len] = sentences[j]
                else:
                    batch_data[j] = sentences[j][:max_len]

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)

            # shift tensors to GPU if available
            batch_data = batch_data.to(self.device)
    
            yield batch_data