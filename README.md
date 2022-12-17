# SimCLS for custom dataset and experiment on paper titling task

## Overview
This repo is a replica of [SimCLS](https://arxiv.org/abs/2106.01890v1) for abstract text summarization. Unlike the original source code, we add some code for generating summary candidates and simplifying the training process. And we also tried this framework with different architecture including deBerta and Bert as scorers for our covid paper titling task.

Lacking of computational power, the generative model we use is a [T5_small ](https://huggingface.co/t5-small) finetuned on our dataset.


## 1. How to Install

### Requirements
- `python3`
- `conda create --name env --file spec-file.txt`
- `pip3 install -r requirements.txt`
- `compare_mt` -> https://github.com/neulab/compare-mt

### Description of Codes
- `main.py` -> training scorer model
- `model.py` -> models
- `data_utils.py` -> dataloader
- `utils.py` -> utility functions
- `preprocess.py` -> data preprocessing
- `generat_cand.py` -> generate candidate summaries for training
- `finetune_model.py` -> finetune your own generative model
- `evaluate_model.py` -> evalualte model with trained scorer

### Workspace
Following directories should be created for our experiments.
- `./cache` -> storing model checkpoints
## 2. Dataset
Need to know that the dataset in this repo [clean_covid.csv](clean_covid.csv) is just a sample dataset only contain 10000 records, if you want to access to the full data, please refer to the following link.

- [The COVID-19 Open Research Dataset.](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-covid-19-open-research?tabs=azure-storage)

## 2. Generating candidates

To generate candidates please run:
```
!python gen_candidate.py --generator_name {args.generator_name} --dataset_name {args.dataset_name} --dataset_percent {args.dataset_percent} --num_cands {args.num_cands}
```
**generator_name**: is the path to previously finetuned generator. Here in our case we use a T5_small model finetuned on CORD dataset.  
**dataset_name**: is the path to dataset. (need to be a csv file, and column name for source document should be **abstract**, column name for reference summary should be **title**). 
**dataset_percent**: percent of data are used to generate, for test you can use smal percent of dataset to debug. Default to 100.  
**num_cands**: Num of candidates you want to generate.  

Generated candidate are stored in the forder 'candidates/{args.generator_name}_{args.num_cands}'.  

For data preprocessing, please run
```
python preprocess.py --src_dir [path of the raw data] --tgt_dir [output path] --split [train/val/test] --cand_num [number of candidate summaries]
```
`src_dir` is the candidate folder: 'candidates/{args.generator_name}_{args.num_cands}'.

The preprocessing precedure will store the processed data as seperate json files in `tgt_dir`.

## 3. scorer training

### Hyper-parameter Setting
You may specify the hyper-parameters in `main.py`.

### Train
```
python main.py --cuda --gpuid [list of gpuid] -l
```
### Fine-tune
```
python main.py --cuda --gpuid [list of gpuid] -l --model_pt [model path]
```
model path should be a subdirectory in the `./cache` directory, e.g. `cnndm/model.pt` (it shouldn't contain the prefix `./cache/`).

### Evaluate
```
python evaluate_model.py --generator_name {args.generator_name} --dataset_name {args.dataset_name} --scorer_path cache/22-12-17-0/scorer.bin --dataset_percent 10
```

## 4. Results

### RoBERTa scorer
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| Before SimCLS | 0.4298  | 0.2275   | 0.3709   |
| After SimCLS  | 0.4105  | 0.1972   | 0.3445   |

### DeBERTa scorer
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| Before SimCLS | 0.4298  | 0.2275   | 0.3709   |
| After SimCLS  | 0.4105  | 0.1972   | 0.3445   |

### BERT scorer
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| Before SimCLS | 0.4298  | 0.2275   | 0.3709   |
| After SimCLS  | 0.4105  | 0.1972   | 0.3445   |
