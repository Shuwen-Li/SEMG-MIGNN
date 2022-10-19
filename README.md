# SEMG-MIGNN
This is a repository for paper "Extrapolative and interpretable reaction performance predic-tion via chemical knowledge-based graph model".

# Introduction

Unlike the current situation of synthetic performance prediction where most studies directly applied available ML algorithms, the innovation of ML framework for molecular property prediction has recently saw remarkable progress. Based on graph representation, a series of breakthroughs of molecular graph models have enabled chemical accuracy-level predictions for millions of small organic molecules, which has generated a strong momentum for artificial intelligence design of functional molecules. These studies revealed the potential of artificial intelligence modelling in chemical research and particularly pointed out the critical role of model framework in chemical ML. In light of the advantage of graph model in representing small organic molecules, we surmise that the SPR model can be innovated by enriching the local encodings of chemical environment and strengthen the information interaction between reaction components. Herein we report a new reaction performance model with two innovative designs; this model for the first time embeds the digitalized steric and electronic information of atomic environment, and the molecular interaction module allows the effective learning of the synergistic control by multiple reaction components. This model achieved excellent yield and stereoselectivity predictions, and our additional experimental tests of asymmetric thiol addition of imines verified its extrapolation ability in new catalyst predictions. This efficient, accurate and explainable model provides a useful approach for reaction performance prediction, which will accelerate the ML design of molecular synthesis.

# Generation of SEMG

Embed steric information:

![SEMG_si_spms](https://user-images.githubusercontent.com/71930017/188142934-59d21532-8a1c-4b48-b879-f395664c6790.jpg)

Embed electronic information

![SEMG_si_ele](https://user-images.githubusercontent.com/71930017/188142893-bbf371ee-1896-49e1-bd28-4976e8c2f9dd.jpg)
# MIGNN

The detailed workflow of the MIGNN model is shown as follows, which uses the chiral phosphoric acid-catalyzed thiol addition to N-acylimines as an example. 

![mignn_si](https://user-images.githubusercontent.com/71930017/188142704-cbf56a26-f2d0-4d69-a768-44d57d6f3f0d.jpg)

# Packages requirements
In order to run Jupyter Notebook involved in this repository, several third-party python packages are required. The versions of these packages in our station are listed below.
```
dgl = 0.7.2
matplotlib = 3.4.2
networkx = 2.6.3
numpy = 1.22.4  
pandas = 1.3.3 
pyscf = 2.0.1
rdkit = 2022.03.2   
scipy = 1.4.1 
seaborn = 0.11.1 
sklearn = 0.23.2  
tensorflow = 2.8.0-dev20211115
xgboost = 1.3.3 
```

# Demo & Instructions for use


Notebook 1 demonstrates how to generate local steric and electronic information in data1.

Notebook 2 demonstrates how to train and predict yield in data1.

Notebook 3 demonstrates how to use generate local steric and electronic information in data2.

Notebook 4 demonstrates how to train and predict enantioselectivity in data2.

Baseline folder demonstrates how to train and predict yield/enantioselectivity by baseline MG-GCN, SEMG-GCN, baseline MG-MIGNN, and classical descriptors-models.


# How to cite
The paper is under review.
# Contact with us
Email: hxchem@zju.edu.cn; shuwen-li@zju.edu.cn
