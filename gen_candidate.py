import torch
import os
import argparse
import numpy as np
import pandas as pd
import datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AutoModel
)
from tabulate import tabulate
import nltk
from datetime import datetime
from datasets import Dataset
import math
import warnings
warnings.filterwarnings("ignore")
import time
from datetime import datetime


def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["abstract"], batch["title"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch

def generate_cands(input_data,args):
    all_cands = []
    cand_gen_batch = args.cand_gen_batch
    num_return_seqs = args.num_cands
    total_doc = list(range(len(input_data)))
    total_batch = math.ceil(len(input_data)/cand_gen_batch)
    used_time = 0
    for batch_index in range(total_batch):
        print(f'current gen batch {batch_index}, total_batch_index: {total_batch}, last epoch time: {used_time} sec', end ='\r')
        time_start = time.time()
        if (batch_index+1)*cand_gen_batch > len(input_data):
            input_ids = train_data.select(total_doc[cand_gen_batch*batch_index:])['input_ids']
            input_ids = torch.tensor(input_ids, dtype=torch.long).to('cuda')
        else:
            input_ids = train_data.select(total_doc[cand_gen_batch*batch_index:(batch_index+1)*cand_gen_batch])['input_ids']
            input_ids = torch.tensor(input_ids, dtype=torch.long).to('cuda')
        train_output_ids = model.generate(input_ids,num_beams = num_return_seqs,
                                 #no_repeat_ngram_size=2,
                                 diversity_penalty=1.0,
                                 max_length = args.decoder_max_len,
                                 num_beam_groups = num_return_seqs,
                                 num_return_sequences=num_return_seqs)
        cands = tokenizer.batch_decode(train_output_ids, skip_special_tokens=True)
        all_cands.extend(cands)
        torch.cuda.empty_cache()
        time_end = time.time()
        used_time = int(time_end-time_start)
    return(all_cands)

def write_data_for_preprocess(tran_test_val, save_dir,doc,refer,cand):
    if os.path.exists(f'{save_dir}/diverse'):
        pass
    else:
        os.makedirs(f'{save_dir}/diverse')
    cand = [line+'\n' for line in cand]
    doc = [line+'\n' for line in doc]
    refer = [line+'\n' for line in refer]
    with open(f'{save_dir}/diverse/{tran_test_val}.out','w') as handle:
        handle.writelines(cand)
    with open(f'{save_dir}/diverse/{tran_test_val}.out.tokenized','w') as handle:
        handle.writelines(cand)
    with open(f'{save_dir}/diverse/{tran_test_val}.source','w') as handle:
        handle.writelines(doc)
    with open(f'{save_dir}/diverse/{tran_test_val}.source.tokenized','w') as handle:
        handle.writelines(doc)
    with open(f'{save_dir}/diverse/{tran_test_val}.target','w') as handle:
        handle.writelines(refer)
    with open(f'{save_dir}/diverse/{tran_test_val}.target.tokenized','w') as handle:
        handle.writelines(refer)

if __name__ == "__main__":
    # need parameter: 1, generator name   2, dataset name  3, decoder max_len  4,num_cands 5, dataset_percent
    parser = argparse.ArgumentParser(description='Candidate generate parameter')
    parser.add_argument("--generator_name", type=str)
    parser.add_argument("--dataset_name", default="clean_covid.csv", type=str)
    parser.add_argument("--decoder_max_len", default=100, type=int)
    parser.add_argument("--num_cands", default=8, type=int)
    parser.add_argument("--dataset_percent", default=100, type=float)
    parser.add_argument("--cand_gen_batch", default=32, type=int)# A100 GPU T5_small can process 32 batch size
    args = parser.parse_args()
    
    
    # load model
    model_name = args.generator_name
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encoder_max_length = 512 # default to 512
    decoder_max_length = args.decoder_max_len

    # read_dataset preprocess data
    csv_data_name = pd.read_csv(args.dataset_name)
    data = csv_data_name[['abstract','title']].dropna()
    dataset = Dataset.from_pandas(data)
    train_data_txt, remain_data_txt = dataset.train_test_split(test_size=0.2,seed = 2333).values()
    val_data_txt, test_data_txt = remain_data_txt.train_test_split(test_size=0.5,seed = 2333).values()
    
    data_percent = args.dataset_percent/100 # for test you can only select part of your dataset for training
    train_data_txt = train_data_txt.shuffle(seed = 2333).select(range(int(len(train_data_txt)*data_percent)))
    val_data_txt = val_data_txt.shuffle(seed = 2333).select(range(int(len(val_data_txt)*data_percent)))
    test_data_txt = test_data_txt.shuffle(seed = 2333).select(range(int(len(test_data_txt)*data_percent)))
    
    print('start tokenize dataset'+'-'*20)
    
    train_data = train_data_txt.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, encoder_max_length, decoder_max_length
        ),
        batched=True,
        remove_columns=train_data_txt.column_names,
    )

    val_data = val_data_txt.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, encoder_max_length, decoder_max_length
        ),
        batched=True,
        remove_columns=val_data_txt.column_names,
    )

    test_data = test_data_txt.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, encoder_max_length, decoder_max_length
        ),
        batched=True,
        remove_columns=test_data_txt.column_names,
    )
    
    
    model.to('cuda')
    
    # start generate candidates
    print('start generate training candidates!' + '-'*50)
    train_cands = generate_cands(train_data, args)
    
    print('start generate validation candidates!' + '-'*50)
    val_cands = generate_cands(val_data, args)
    
    print('start generate testing candidates!' + '-'*50)
    test_cands = generate_cands(test_data, args)
    
    
    print('start saving candidate summaries')
    
    write_data_for_preprocess('val',f'candidates/{args.generator_name}_{args.num_cands}',
                              val_data_txt['abstract'],val_data_txt['title'],val_cands)

    write_data_for_preprocess('train',f'candidates/{args.generator_name}_{args.num_cands}',
                              train_data_txt['abstract'],train_data_txt['title'],train_cands)

    write_data_for_preprocess('test',f'candidates/{args.generator_name}_{args.num_cands}',
                              test_data_txt['abstract'],test_data_txt['title'],test_cands)
