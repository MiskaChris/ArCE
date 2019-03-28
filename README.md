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
Tasks       |                   Prediction Head (Hits@10)|            Prediction Tail (Hits@10)
:------------------       | :---------------:      | :------------------: 

Relation Category | 1-to-1 | 1-to-n | n-to-1 |m-to-n | 1-to-1 | 1-to-n | n-to-1 |m-to-n |
:--------- | :---: | :---: | :---: | :---: | :---:| :---: | :---: | :---: 
TransE(Bodes et al., 2013) | 43.7| 65.7 | 18.2 | 47.2 | 43.7 | 19.7 | 66.7 | 50.0 
![Image text](https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/canon.png)


