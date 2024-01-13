"""Evaluate the model"""

import argparse
import random
import logging
import os

import numpy as np
import torch

from pytorch_pretrained_bert import BertForTokenClassification, BertConfig

from metrics import f1_score
from metrics import classification_report

from data_loader import DataLoader
from transformers import AutoTokenizer
import utils

import itertools
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

def max_sum_distance(
        doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: List[str],
        top_n: int,
        nr_candidates: int,
    ) -> List[Tuple[str, float]]:
    """Calculate Max Sum Distance for extraction of keywords

    We take the 2 x top_n most similar words/phrases to the document.
    Then, we take all top_n combinations from the 2 x top_n words and
    extract the combination that are the least similar to each other
    by cosine similarity.

    This is O(n^2) and therefore not advised if you use a large `top_n`

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        nr_candidates: The number of candidates to consider

    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances
    """
    if nr_candidates < top_n:
        raise Exception(
            "Make sure that the number of candidates exceeds the number "
            "of keywords to return."
        )
    elif top_n > len(words):
        return []

    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, word_embeddings)
    distances_words = cosine_similarity(word_embeddings, word_embeddings)

    # Get 2*top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [words[index] for index in words_idx]
    candidates = distances_words[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = 100_000
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum(
            [candidates[i][j] for i in combination for j in combination if i != j]
        )
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [
        (words_vals[idx], round(float(distances[0][words_idx[idx]]), 4))
        for idx in candidate
    ]

def split_list_into_chunks(input_list, chunk_size):
    # Split the list into chunks
    chunks = [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
    
    # Check if the last chunk needs to be filled with zeros
    if len(chunks[-1]) < chunk_size:
        chunks[-1].extend([0] * (chunk_size - len(chunks[-1])))
    
    return chunks


def sample_test(sample_document):
    return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model_dir', default='pretrained/scibert_scivocab_uncased', help="Directory containing the BERT model in PyTorch")
    parser.add_argument('--model_dir', default='experiments/teacher_model', help="Directory containing params.json")
    parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")
    parser.add_argument('--restore_file', default='best', help="name of the file in `model_dir` containing weights to load")
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

    f = open("sample.txt", "r")
    sample_document = f.read()

    tag2idx = {"I":0, "O":1}
    idx2tag = {0: "I", 1: "0"}

    # Define the model
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)
    config_path = os.path.join(args.bert_model_dir, 'config.json')
    config = BertConfig.from_json_file(config_path)
    model = BertForTokenClassification(config, num_labels=2)

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    model.to(params.device)
    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    model.eval()
    full_output = torch.empty(0, params.max_len, 2, device=params.device)
    # fetch the next evaluation batch
    sample_document = sample_document.lower()
    sample_tokens = tokenizer.encode(sample_document)
    full_data = split_list_into_chunks(sample_tokens, params.max_len)
    full_data = torch.tensor(full_data, dtype=torch.long).to(params.device)
    max_batch = 4
    print("Start inference")
    for i in range(0, full_data.shape[0], max_batch):
        batch_data = full_data[i:i+max_batch]
        batch_output = model(batch_data, token_type_ids=None, attention_mask=batch_data.gt(0))  # shape: (batch_size, max_len, num_labels)
        full_output = torch.cat((full_output, batch_output), dim=0)
    full_output = full_output.detach().cpu().numpy()

    output_tag = [idx2tag.get(idx) for indices in np.argmax(full_output, axis=2) for idx in indices]
    output_document = [tokenizer.convert_ids_to_tokens(word) for word in sample_tokens]
    keywords = [""]
    for i in range(len(output_document)):
        if output_tag[i] == 'I':
            keywords[-1] = (keywords[-1] + " " + output_document[i]).strip()
        if output_tag[i] == '0' and keywords[-1]!="":
            keywords.append("")
    keywords.pop()

    ## After get a list of candidate, use Sentence embedding to find the top_n keywords
    doc_embedding = sentence_model.encode(sample_document)
    candidate_embeddings = [np.array(sentence_model.encode(kw)) for kw in keywords]

    # result = max_sum_distance(doc_embedding, word_embeddings, keywords, 20, len(word_embeddings))
    top_n = 10
    #Find the top N candidates with closest cosine distance.
    similarities = [cosine_similarity(doc_embedding.reshape(1,-1), candidate.reshape(1,-1)) for candidate in candidate_embeddings]
    # Get the top N indices sorted by highest similarity
    indexed_scores = [(i, array.item()) for i, array in enumerate(similarities)]

    # Sort based on scores in descending order
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the indices of the top top_n scores
    top_indices = [index for index, score in indexed_scores[:top_n]]
    seen = set()
    result = [keywords[idx] for idx in top_indices if not (keywords[idx] in seen or seen.add(keywords[idx]))]
    print(result)