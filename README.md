# JLSD implement

This is an attempt to reimplement JLSD technique, taken from this paper: A Joint Learning Approach based on Self-Distillation for Keyphrase Extraction from Scientific Documents (Tuan Manh Lai, Trung Bui, Doo Soon Kim, Quan Hung Tran)

## Usage

1. `python -m pip install -r requirements.txt`
2. From `scibert` repo, untar the weights (rename their weight dump file to `pytorch_model.bin`) and vocab file into a new folder `model`.
3. Change the parameters accordingly in `experiments/base_model/params.json`. We recommend keeping batch size of 4 and sequence length of 512, with 6 epochs, if GPU's VRAM is around 11 GB.
4. Check script.sh for training and testing command

### Todo

We only considered a linear layer on top of BERT embeddings. We need to see whether SciBERT + BiLSTM + CRF makes a difference.

## Credits

1. SciBERT: https://github.com/allenai/scibert
2. HuggingFace: https://github.com/huggingface/pytorch-pretrained-BERT
3. PyTorch NER: https://github.com/lemonhu/NER-BERT-pytorch
4. BERT: https://github.com/google-research/bert

## Reference
https://github.com/pranav-ust/BERT-keyphrase-extraction
