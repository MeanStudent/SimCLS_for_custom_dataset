import torch
import numpy as np
import pandas as pd
import datasets
import argparse
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AutoModel,
    RobertaModel, 
    RobertaTokenizer
)
from tabulate import tabulate
import model
import nltk
from datetime import datetime
from datasets import Dataset
import math
import warnings
warnings.filterwarnings("ignore")
import time
from data_utils import to_cuda, collate_mp, ReRankingDataset
from torch.utils.data import DataLoader
from compare_mt.rouge.rouge_scorer import RougeScorer

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
        [-100 if token == generator_tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch
def to_tensor(ids):
    return torch.tensor(ids, dtype=torch.long).to('cuda')

def evaluate_without_SimCLS(generator, test_data_txt, cand_num, show_results):
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    doc_txt = test_data_txt['abstract']
    doc_ids = generator_tokenizer.batch_encode_plus(doc_txt, max_length = 512, pad_to_max_length=True)['input_ids']
    doc_ids = to_tensor(doc_ids)
    doc_input_mask = doc_ids != scorer_tokenizer.pad_token_id
    doc_out = scorer.encoder(doc_ids, attention_mask=doc_input_mask)['last_hidden_state']
    doc_emb = torch.mean(doc_out,dim = 1) # average over all word embeddings
    
    ref_txt = test_data_txt['title']
    ref_ids = generator_tokenizer.batch_encode_plus(ref_txt, max_length = 512, pad_to_max_length=True)['input_ids']
    ref_ids = to_tensor(ref_ids)
    ref_input_mask = ref_ids != scorer_tokenizer.pad_token_id
    ref_out = scorer.encoder(ref_ids, attention_mask=ref_input_mask)['last_hidden_state']
    ref_emb = torch.mean(ref_out,dim = 1) 
    
    cand_id = generator.generate(doc_ids,num_beams = 16,
                                     #no_repeat_ngram_size=2,
                                     diversity_penalty=1.0,
                                     max_length = 20,
                                     num_beam_groups = cand_num,
                                     num_return_sequences = 1)
    cands_txt = generator_tokenizer.batch_decode(cand_id, skip_special_tokens=True)
    candidate_id = cand_id.view(-1, cand_id.size(-1))
    cand_input_mask = candidate_id != scorer_tokenizer.pad_token_id
    cand_out = scorer.encoder(candidate_id, attention_mask=cand_input_mask)['last_hidden_state'] 
    candidate_embs = torch.mean(cand_out,dim = 1)
    
    cand_similarity_score = torch.cosine_similarity(candidate_embs, doc_emb, dim=-1).item()
    ref_similarity_score = torch.cosine_similarity(ref_emb, doc_emb, dim=-1).item()
    
    cands_rouge_scores = rouge_scorer.score(cands_txt[0],ref_txt[0])
    rouge1_scores = cands_rouge_scores['rouge1'].fmeasure
    rouge2_scores = cands_rouge_scores['rouge2'].fmeasure
    rougeL_scores = cands_rouge_scores['rougeLsum'].fmeasure
    
    if show_results:
        print('doc'+'-'*50)
        print(doc_txt)
        print('ref'+'-'*50)
        print(ref_txt)
        print('cand'+'-'*49)
        print(cands_txt)
        print('scores:'+'-'*50)
        print(f'rouge1: {rouge1_scores}, rouge2: {rouge2_scores}, rougeL: {rougeL_scores}')
        print(f'cand similarity: {cand_similarity_score}, ref similarity: {ref_similarity_score}')
    regular_scores = {'rouge1': rouge1_scores, 
                    'rouge2': rouge2_scores,
                    'rougeL': rougeL_scores,
                    'similar': cand_similarity_score,
                    'ref_similar':ref_similarity_score}
    
    return regular_scores,cands_txt

def evaluate_SimCLS(generator, generator_tokenizer, scorer, scorer_tokenizer, 
                    test_data_txt, cand_num, show_piece_of_data):
    # generate batch data
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    # 1, encode doc
    doc_txt = test_data_txt['abstract']
    doc_ids = generator_tokenizer.batch_encode_plus(doc_txt, max_length = 512, pad_to_max_length=True)['input_ids']
    doc_ids = to_tensor(doc_ids)
    # 2, encode true sum
    ref_txt = test_data_txt['title']
    ref_ids = generator_tokenizer.batch_encode_plus(ref_txt, max_length = 512, pad_to_max_length=True)['input_ids']
    ref_ids = to_tensor(ref_ids)
    # 3, generate cands
    cands_ids = generator.generate(doc_ids,num_beams = cand_num,
                                     #no_repeat_ngram_size=2,
                                     diversity_penalty=1.0,
                                     max_length = 20,
                                     num_beam_groups = cand_num,
                                     num_return_sequences = cand_num)
    cands_txt = generator_tokenizer.batch_decode(cands_ids, skip_special_tokens=True)
    # 4, get sentence embeddings
        # doc emb
    doc_input_mask = doc_ids != scorer_tokenizer.pad_token_id
    doc_out = scorer.encoder(doc_ids, attention_mask=doc_input_mask)['last_hidden_state']
    doc_emb = torch.mean(doc_out,dim = 1) # average over all word embeddings
        # cands emb
    candidate_id = cands_ids.view(-1, cands_ids.size(-1))
    cand_input_mask = candidate_id != scorer_tokenizer.pad_token_id
    cand_out = scorer.encoder(candidate_id, attention_mask=cand_input_mask)['last_hidden_state'] 
    candidate_embs = torch.mean(cand_out,dim = 1)
        # ref emb
    ref_input_mask = ref_ids != scorer_tokenizer.pad_token_id
    ref_out = scorer.encoder(ref_ids, attention_mask=ref_input_mask)['last_hidden_state']
    ref_emb = torch.mean(ref_out,dim = 1) 
    
    similarity_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    for i in range(cand_num):
        score = torch.cosine_similarity(candidate_embs[i], doc_emb, dim=-1).item()
        similarity_scores.append(score)
        cands_rouge_scores = rouge_scorer.score(cands_txt[i],ref_txt[0])
        rouge1_scores.append(cands_rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(cands_rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(cands_rouge_scores['rougeLsum'].fmeasure)
        
    ref_similarity_score = torch.cosine_similarity(ref_emb, doc_emb, dim=-1).item()

    if show_piece_of_data:
        print('-'*50)
        show_a_piece_of_data(generator, generator_tokenizer, test_data_txt)
    max_index = similarity_scores.index(max(similarity_scores))
    top1_scores = {'rouge1': rouge1_scores[max_index], 
                    'rouge2': rouge2_scores[max_index],
                    'rougeL': rougeL_scores[max_index],
                    'similar': similarity_scores[max_index],
                    'ref_similar': ref_similarity_score}
    
    return top1_scores, cands_txt, max_index

if __name__ == '__main__':
    #parameters needed: 1, generator name 2, scorer name 3, dataset name 4, gen_candidate 5, max_len
    parser = argparse.ArgumentParser(description='Candidate generate parameter')
    parser.add_argument("--generator_name", type=str)
    parser.add_argument("--scorer_path", type = str)
    parser.add_argument("--dataset_name", default="clean_covid.csv", type=str)
    parser.add_argument("--gen_length", default=100, type=int)
    parser.add_argument("--num_cands", default=8, type=int)
    parser.add_argument("--dataset_percent", default=100, type=float)
    parser.add_argument("--scorer_architecture_name", default='roberta-base', type=str)
    parser.add_argument("--cand_gen_batch", default=32, type=int)# A100 GPU T5_small can process 32 batch size
    parser.add_argument("--evaluate_batch_size", default=32, type=int)
    args = parser.parse_args()
    device = 'cuda'
    
    model_name = args.generator_name
    generator = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    generator_tokenizer = AutoTokenizer.from_pretrained(model_name)

    encoder_max_length = 512 # default to 512
    decoder_max_length = args.gen_length

    # read_dataset preprocess data
    csv_data_name = pd.read_csv(args.dataset_name)
    data = csv_data_name[['abstract','title']].dropna()
    dataset = Dataset.from_pandas(data)
    train_data_txt, remain_data_txt = dataset.train_test_split(test_size=0.2,seed = 2333).values()
    val_data_txt, test_data_txt = remain_data_txt.train_test_split(test_size=0.5,seed = 2333).values()
    # Sinice we are evaluating, we only need test set 
    percent_data = args.dataset_percent/100
    test_data_txt = test_data_txt.shuffle(seed = 2333).select(range(int(len(test_data_txt)*percent_data)))
    print('start tokenize data')
    test_data = test_data_txt.map(
        lambda batch: batch_tokenize_preprocess(
            batch, generator_tokenizer, encoder_max_length, decoder_max_length
        ),
        batched=True,
        remove_columns=test_data_txt.column_names,
    )
    print(f'There are {len(test_data)} samples in test set!')
    generator.to(device)
    
    # load scorer_tokenizer
    print('start loading scorer model')
    scorer_name = args.scorer_architecture_name
    pt_scorer_path = args.scorer_path
    scorer_tokenizer = RobertaTokenizer.from_pretrained(scorer_name)
    scorer = model.ReRanker(scorer_name, scorer_tokenizer.pad_token_id)
    scorer.load_state_dict(torch.load(pt_scorer_path,map_location=torch.device(device)))
    scorer.to(device)
    
    print(f'start evaluating, total_num of samples is:{len(test_data_txt)}')
    # evaluate loop:
    rouge1_noSimCLS = []
    rouge2_noSimCLS = []
    rougeL_noSimCLS = []

    rouge1_SimCLS = []
    rouge2_SimCLS = []
    rougeL_SimCLS = []

    references = []
    regular_cand_pred = []
    SimCLS_cand_pred = []

    for i in range(0,len(test_data_txt)):
        time_start = time.time()

        regular_scores,cand_txt = evaluate_without_SimCLS(
            generator, test_data_txt[i:i+1], args.num_cands, False)

        rouge1_noSimCLS.append(regular_scores['rouge1'])
        rouge2_noSimCLS.append(regular_scores['rouge2'])
        rougeL_noSimCLS.append(regular_scores['rougeL'])


        SimCLS_sores, cands_txt, top1_index = evaluate_SimCLS(generator, generator_tokenizer, scorer, 
                        scorer_tokenizer, test_data_txt[i:i+1], args.num_cands, False)

        rouge1_SimCLS.append(SimCLS_sores['rouge1'])
        rouge2_SimCLS.append(SimCLS_sores['rouge2'])
        rougeL_SimCLS.append(SimCLS_sores['rougeL'])

        references.append(dataset[i]['title'])
        regular_cand_pred.append(cand_txt[0])
        SimCLS_cand_pred.append(cands_txt[top1_index])
        time_end = time.time()

        time_used = time_end - time_start
        print(f'current working sample: {i+1}, time used last sample: {round(time_used,4)}', end = '\r')
       
    print('')
    print(f'Before SimCLS ROUGE1: {np.mean(rouge1_noSimCLS)}, ROUGE2: {np.mean(rouge2_noSimCLS)}, ROUGEL: {np.mean(rougeL_noSimCLS)}')
    print(f'After SimCLS ROUGE1: {np.mean(rouge1_SimCLS)}, ROUGE2: {np.mean(rouge2_SimCLS)}, ROUGEL: {np.mean(rougeL_SimCLS)}')
