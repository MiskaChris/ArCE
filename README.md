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

## Supplementary Experiments1 
To further analyze the KGE results, researchers often classify a KG’s relations into 4 types according to the type definition given by TransH and then report more detailed experimental results of them. We also compare the detailed results of ArcE(Basic) with other baselines.   
The results are shown in Table 4. From Table 4 we can see that ArcE(Basic) obtains more balanced performance among different kinds of relations: in ArcE(Basic), the performance gaps among different kinds of relations are less than that of in other methods. Especially, ArcE(Basic) does much better than other baselines for the m-to-n relations.  
From Table 4 we can also find that when predicting the complex part of a relation, most of methods do much poor. For example, the Hits@10 scores for both the n-to-1 relations’ head prediction and the 1-to-n relations’ tail prediction are far lower than the scores of other kinds of prediction. This trend also exists in ArcE(Basic), but our method does much better than the compared baseline methods, which indicates our method can alleviate the data sparisity issue greatly. 
<div align=center><img width="800" height="300" src="https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C.png"/></div>
Table 4: Experimental results of FB15k by mapping properties of relations. Here we only compare ArcE with
some of the baselines that report this kind of experimental results. Especially compared with TransG, which
achieves the current-best Hits@10 score on FB15k.  

## Supplementary Experiments2
 Here r refers to the atrous rate. Atrous means the number of atrous layers in the AtrousConvBlock.  
 1. Hits@3 results under different setting are shown below.
 <div align=center><img width="600" height="200" src="https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/实验1.png"/></div>  
 2.  Hits@1 results under different setting are shown below.  
  <div align=center><img width="800" height="300" src="https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/实验2.png"/></div>   
  
  3.  MRR results under different setting are shown below.  
  <div align=center><img width="800" height="300" src="https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/实验3.png"/></div> 




