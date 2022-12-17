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
**dataset_name**: is the path to dataset. (need to be a csv file, and column name for source document should be **abstract**, column name for reference summary should be **title**)
**dataset_percent**: percent of data are used to generate, for test you can use smal percent of dataset to debug. Default to 100.
**num_cands**: Num of candidates you want to generate.

Generated candidate are stored in the forder 'candidates/{args.generator_name}_{args.num_cands}'

For data preprocessing, please run
```
python preprocess.py --src_dir [path of the raw data] --tgt_dir [output path] --split [train/val/test] --cand_num [number of candidate summaries]
```
`src_dir` is the candidate folder: 'candidates/{args.generator_name}_{args.num_cands}'.

Each line of these files should contain a sample. In particular, you should put the candidate summaries for one data sample at neighboring lines in `test.out` and `test.out.tokenized`.

The preprocessing precedure will store the processed data as seperate json files in `tgt_dir`.

## 3. How to Run

### Hyper-parameter Setting
You may specify the hyper-parameters in `main.py`.

To reproduce our results, you could use the original configuration in the file, except that you should make sure that on CNNDM 
`args.max_len=120`, and on XSum `args.max_len = 80`.


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
python main.py --cuda --gpuid [single gpu] -e --model_pt [model path]
```
model path should be a subdirectory in the `./cache` directory, e.g. `cnndm/model.pt` (it shouldn't contain the prefix `./cache/`).

## 4. Results

### CNNDM
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| BART     | 44.39   | 21.21   | 41.28   |
| Ours     | 46.67   | 22.15   | 43.54   |

### XSum
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| Pegasus  | 47.10   | 24.53   | 39.23   |
| Ours     | 47.61   | 24.57   | 39.44   |

Our model outputs on these datasets can be found in `./output`.

We have also provided the finetuned checkpoints on [CNNDM](https://drive.google.com/file/d/1CSFeZUUVFF4ComY6LgYwBpQJtqMgGllI/view?usp=sharing) and [XSum](https://drive.google.com/file/d/1yx9KhDY0CY8bLdYnQ9XhvfMwxoJ4Fz6N/view?usp=sharing).
