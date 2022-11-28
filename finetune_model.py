import os
os.environ["WANDB_DISABLED"] = "true"
import torch
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
import argparse
from datetime import datetime
from datasets import Dataset
# Define functions --------------------------------------------------------------------------------------------------------
def batch_tokenize_preprocess(args,batch, tokenizer, max_source_length, max_target_length):
    source, target = batch[args.doc_col_name], batch[args.sum_col_name]
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

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# Process datasets --------------------------------------------------------------------------------------------------------
def preprocess_data(args):
    dataset = pd.read_csv(args.data_file)
    dataset = dataset[[args.doc_col_name,args.sum_col_name]].dropna()
    dataset = Dataset.from_pandas(dataset)

    train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.2).values()
    encoder_max_length = 512  # demo
    decoder_max_length = 100

    train_data = train_data_txt.map(
        lambda batch: batch_tokenize_preprocess(
            args, batch, tokenizer, encoder_max_length, decoder_max_length
        ),
        batched=True,
        remove_columns=train_data_txt.column_names,
    )

    validation_data = validation_data_txt.map(
        lambda batch: batch_tokenize_preprocess(
            args, batch, tokenizer, encoder_max_length, decoder_max_length
        ),
        batched=True,
        remove_columns=validation_data_txt.column_names,
    )
    return train_data,validation_data

# Start finetuning model --------------------------------------------------------------------------------------------------
def train_and_save(args, train_data, validation_data):
    training_args = Seq2SeqTrainingArguments(
        output_dir=f'{args.save_dir}/{args.pt_model}/output',
        num_train_epochs=args.num_epoch,  # demo
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,  # demo
        per_device_eval_batch_size=args.batch_size,
        # learning_rate=3e-05,
        warmup_steps=500,
        weight_decay=0.1,
        label_smoothing_factor=0.1,
        predict_with_generate=True,
        logging_dir=f'{args.save_dir}/{args.pt_model}/logs',
        logging_steps=50,
        save_total_limit=3,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    if os.path.exists(f'{args.save_dir}/{args.pt_model}'):
        print('trained model folders already exist, creating new dir')
        dtime_str = str(time.asctime(time.localtime(time.time()))).replace(' ','_')
        os.makedirs(f'{args.save_dir}/{args.pt_model}_{dtime_str}')
        new_path = f'{args.save_dir}/{args.pt_model}_{dtime_str}'
    else:
        os.makedirs(f'{args.save_dir}/{args.pt_model}')
        new_path = f'{args.save_dir}/{args.pt_model}'

    trainer.save(new_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing Parameter')
    parser.add_argument("--pt_model", type=str)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--doc_col_name", type=str)
    parser.add_argument("--sum_col_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    
    nltk.download("punkt", quiet=True)
    model_name = args.pt_model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    metric = datasets.load_metric("rouge")
    
    #load data
    print('start loading and preprocess data-------------------------')
    train_data,validation_data = preprocess_data(args)
    
    #start training
    print('start finetuning the model-------------------------')
    train_and_save(args, train_data, validation_data)
    
    print('finished!')
    

