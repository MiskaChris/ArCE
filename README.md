# basic_ArcE



## Installation

This repo supports Linux and Python installation via Anaconda. 

1. Install [PyTorch](https://github.com/pytorch/pytorch) using [Anaconda](https://www.continuum.io/downloads). 
2. Install the requirements `pip install -r requirements`
3. Download the default English model used by [spaCy](https://github.com/explosion/spaCy), which is installed in the previous step `python -m spacy download en`
4. Preprocess for FB15k : `python wrangle_KG.py FB15k`
5. You can now run the model

## Running a model

Parameters need to be specified by white-space tuples for example:
```

CUDA_VISIBLE_DEVICES=3 python main.py model ArcE dataset FB15k input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 process True 
```
will run a basic_ArcE model on FB15k which will not utilize the descriptions of the entities.

----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------

# boosting_ArcE

## Installation

This repo supports Linux and Python installation via Anaconda. 

1. Install [PyTorch](https://github.com/pytorch/pytorch) using [Anaconda](https://www.continuum.io/downloads). 
2. Install the requirements `pip install -r requirements`
3. Download the default English model used by [spaCy](https://github.com/explosion/spaCy), which is installed in the previous step `python -m spacy download en`
4. Preprocess for FB15k : `python wrangle_KG.py FB15k`
5. You can  get the embeddings of description from here : [download](https://pan.baidu.com/s/18N6VkWIm5jeO0MMZMk-4yA),extraction code is `duvb`
6. You can now run the model

## Running a model

Parameters need to be specified by white-space tuples for example:
```

CUDA_VISIBLE_DEVICES=3 python main.py model ArcE dataset FB15k input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 process True 
```
will run a boosting_ArcE model on FB15k which will utilize the descriptions of the entities.

## Supplementary Experiments
Tasks       | hits@3 |    Hits@1
:---        | :---:      | :---: 
Dataset | MR | MRR | Hits@10 | Hits@3 | Hits@1
:--- | :---: | :---: | :---: | :---: | :---:
FB15k | 64 | 0.75 | 0.87 | 0.80 | 0.67
WN18 | 504 | 0.94 | 0.96 | 0.95 | 0.94
FB15k-237 | 246 | 0.32 | 0.49 | 0.35 | 0.24
WN18RR | 4766 | 0.43 | 0.51 | 0.44 | 0.39
YAGO3-10 | 2792 | 0.52 | 0.66 | 0.56 | 0.45
Nations | 2 | 0.82 | 1.00 | 0.88 | 0.72
UMLS | 1 | 0.94 | 0.99 | 0.97 | 0.92
Kinship | 2 | 0.83 | 0.98 | 0.91 | 0.73
