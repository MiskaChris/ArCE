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
<div align=center><img width="700" height="200" src="https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/实验4.png"/></div>
Link prediction results on WN18 & FB15k. (RotatE: https://arxiv.org/pdf/1902.10197.pdf )  
<div align=center><img width="800" height="200" src="https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/实验5.png"/></div>
Link prediction results on WN18RR & FB15k-237. (RotatE: https://arxiv.org/pdf/1902.10197.pdf )  

## Supplementary Experiments3
 
 **Parameter size comparison.**
 <div align=center><img width="800" height="205" src="https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/实验6.png"/></div> 
 Here we copy the qualitative analyzed results for the first two models from ProjE directly(see Table 1 in ProjE for detailed information). ne, nr, nw, are the number of entities, relationships, and words. We use a unified symbol d to denote the dimension of entity embedding, relation embedding, and word embedding. k is the number of hidden layers, h is the size of hidden layers. Usually, nwkh and nekh. The number “2” in DKRL means there are two kinds of matrix parameters: one is for the head entities and the other is for the tail entities. The number “2” in AcrE(Basic) has a similar meaning. The number “2” in AcrE(Full) means there are two kinds of descriptions used. The number “4” means 4 kinds of matrix parameters: the head entity matrix parameters and the tail entity matrix parameters from two kinds of descriptions.  
 
The parameters of our method mainly come from 3 parts.1)Embeddings for entities and relations. All KGE methods involve these parameters because the aim of a KGE method is to obtain good embedding representations for entities and relations. We denote the number of these parameters as d1*(|E|+|R|),d1 is the embedding dimension, E&R are entity set and relation set respectively. 2)Word embeddings used in description representation learning. All description-used methods involve these parameters. We denote the number of these parameters as d2*|W|, d2 is the word embedding dimension, W is the word set. 3)Matrix parameters in our AtrousConvBlock. We denote the number of these parameters as k*|M|, k is the number of network layers(including both the standard convolution layers and the atrous convolution layers), M is transformation matrices. Generally, the sizes of the former two kinds of parameters are far larger than the third one. For example, WN18 has 18 relations and 40943 entities. For word embeddings, the number of words is usually hundreds of thousands. For example, there are 130000 words in SENNA embeddings (Collobert et al.,2011).Embedding dimension is usually set to several hundreds. Thus, there maybe several millions of parameters for either word or entity/relation embeddings. But the number of the third kind of parameters is far smaller. For example,M2,one of the matrices in our method (Equation10 in page4),is set to 100*200. Other matrices have similar parameter numbers. k(the number of network layers) is usually less than 10. Thus, d1*(|E|+|R|)>>k*|M| and d2*|W|>>>k*|M|. From the table we can see that when we set the dimension of embeddings to 200, AcrE(Basic) has a similar number of parameters with ConvE, which is in line with our expectation: the atrous convolution model almost wouldn’t increase the number of parameters. On the other hand, the number of parameters in AcrE(Full) is almost double that of in AcrE(Basic). This shows that the number of parameters in a description-used KGE model mainly come from the embedding parts.
Compared with other description-used methods, the number of parameters in AcrE is linear increased due to the using of 2 kinds descriptions. When modeling these descriptions, DNN-based methods are often used. Thus the differences in both the time complexity and the space complexity of the description-used methods mainly come from the number of parameters. Accordingly, AcrE has a similar time and space complexity with other description-used methods like TKRL,DKRL,SSP,etc.  

## Supplementary Experiments4  
 <div align=center><img width="790" height="160" src="https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/实验7.png"/></div>
 Ablation experiments with/without residual learning on Wn18& FB15k 
   
   
  <div align=center><img width="800" height="160" src="https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/实验8.png"/></div>
  Ablation experiments with/without residual learning on Wn18RR& FB15k-23  
  
## Supplementary Experiments5
**Here r refers to the atrous rate. Atrous means the number of atrous layers in the AtrousConvBlock.**  
 1. Hits@3 results under different setting are shown below.
 <div align=center><img width="630" height="250" src="https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/实验1.png"/></div>  
 2.  Hits@1 results under different setting are shown below.  
  <div align=center><img width="630" height="250" src="https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/实验2.png"/></div>   
  
  3.  MRR results under different setting are shown below.  
  <div align=center><img width="600" height="250" src="https://github.com/MiskaChris/ArCE/blob/master/ArcE/basic_ArcE/实验3.png"/></div>  
  
