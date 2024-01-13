import random
import logging
import os

import numpy as np
import torch

from pytorch_pretrained_bert import BertForTokenClassification, BertConfig

from transformers import AutoTokenizer

import itertools
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from collections import Counter

def split_list_into_chunks(input_list, chunk_size):
    # Split the list into chunks
    chunks = [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
    
    # Check if the last chunk needs to be filled with zeros
    if len(chunks[-1]) < chunk_size:
        chunks[-1].extend([0] * (chunk_size - len(chunks[-1])))
    
    return chunks

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


class KeywordExtractor:
    def __init__(self, model_dir=''):
        """
        Initializes the keyword extractor with the specified model.
        :param model_dir: Directory of the model to load.
        """

        # Use GPUs if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_gpu = torch.cuda.device_count()

        self.max_len = 512
        # Set the random seed for reproducible experiments
        seed = 23
        random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)  # set random seed for all GPUs

        self.idx2tag = {0: "I", 1: "0"}

        # Define the model
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)
        config_path = os.path.join("/colab/jlsd/pretrained/scibert_scivocab_uncased", 'config.json')
        config = BertConfig.from_json_file(config_path)
        self.model = BertForTokenClassification(config, num_labels=2)
        self.model.to(self.device)
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # Reload weights from the saved file
        load_checkpoint(os.path.join('/colab/jlsd/experiments/base_model', 'best.pth.tar'), self.model)
        self.model.eval()

    def inference(self, text, top_n=10):
        """
        Extracts keywords from the given text.
        :param text: The text to extract keywords from.
        :param top_n: Number of top keywords to return.
        :return: A list of top_n keywords with score
        """
        self.model.eval()
        full_output = []
        # Process the text # fetch the next evaluation batch
        sample_document = text.lower()
        print("Start inference")
        sample_tokens = self.tokenizer.encode(sample_document)
        full_data = split_list_into_chunks(sample_tokens, self.max_len)
        max_batch = 4
        for i in range(0, len(full_data), max_batch):
            batch_data = torch.tensor(full_data[i:i+max_batch], dtype=torch.int).to(self.device)
            batch_output = self.model(batch_data, token_type_ids=None, attention_mask=batch_data.gt(0))  # shape: (batch_size, max_len, num_labels)
            full_output.append(batch_output.detach())
            torch.cuda.empty_cache()
        full_output = torch.cat(full_output, dim=0).cpu().numpy()

        output_tag = [self.idx2tag.get(idx) for indices in np.argmax(full_output, axis=2) for idx in indices]
        output_document = [self.tokenizer.convert_ids_to_tokens(word) for word in sample_tokens]
        keywords = [""]
        for i in range(len(output_document)):
            if output_tag[i] == 'I':
                keywords[-1] = (keywords[-1] + " " + output_document[i]).strip()
            if output_tag[i] == '0' and keywords[-1]!="":
                keywords.append("")
        keywords.pop()

        ## After get a list of candidate, use Sentence embedding to find the top_n keywords
        doc_embedding = self.sentence_model.encode(sample_document)
        candidate_embeddings = [np.array(self.sentence_model.encode(kw)) for kw in keywords]

        #Find the top N candidates with closest cosine distance.
        similarities = [cosine_similarity(doc_embedding.reshape(1,-1), candidate.reshape(1,-1)) for candidate in candidate_embeddings]
        # Get the top N indices sorted by highest similarity
        indexed_scores = [(i, array.item()) for i, array in enumerate(similarities)]

        # Sort based on scores in descending order
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Get the indices of the top top_n scores
        seen = set()
        result = [(keywords[idx],score) for idx, score in indexed_scores if not (keywords[idx] in seen or seen.add(keywords[idx]))]
        result = result[:top_n]
        # Return the most common keywords
        return result

if __name__=="__main__":
    # Example usage
    extractor = KeywordExtractor()
    f = open("/colab/jlsd/sample.txt", "r")
    sample_document = f.read()
    
    keywords = extractor.inference(sample_document)
    print(keywords)


   