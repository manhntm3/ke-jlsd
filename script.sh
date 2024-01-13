### First install the needed package
python -m pip install -r requirements.txt

#### Training teacher model and then train join learning 
#Inspec
python train.py --data_dir dataset/Inspec --bert_model_dir pretrained/scibert_scivocab_uncased --model_dir experiments/teacher_model
python evaluate.py --data_dir dataset/Inspec --bert_model_dir pretrained/scibert_scivocab_uncased --model_dir experiments/teacher_model

python train_jlsd.py --data_dir dataset/Inspec --bert_model_dir pretrained/scibert_scivocab_uncased --model_dir experiments/base_model 

#SemEval17
python train.py --data_dir dataset/SemEval17 --bert_model_dir pretrained/scibert_scivocab_uncased --model_dir experiments/teacher_model
python evaluate.py --data_dir dataset/SemEval17 --bert_model_dir pretrained/scibert_scivocab_uncased --model_dir experiments/teacher_model

python train_jlsd.py --data_dir dataset/SemEval17 --bert_model_dir pretrained/scibert_scivocab_uncased --model_dir experiments/base_model 