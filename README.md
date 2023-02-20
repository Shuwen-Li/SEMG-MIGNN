# SEMG-MIGNN
This is a repository for paper "Extrapolative and interpretable reaction performance prediction via chemical knowledge-based graph model".

# Abstract

Accurate prediction of reactivity and selectivity provides the desired guideline for synthetic development. Due to the high-dimensional relationship between molecular structure and synthetic function, it is challenging to achieve the predictive modelling of synthetic transformation with the required extrapolative ability and chemical interpretability. To meet the gap between the rich domain knowledge of chemistry and the advanced molecular graph model, herein we report a new knowledge-based graph model that embeds the digitalized steric and electronic information. In addition, a molecular interaction module is developed to enable the learning of the synergistic influence of reaction components. This knowledge-based graph model achieved excellent predictions of reaction yield and stereoselectivity, whose extrapolative ability was corroborated by additional scaffold-based data splittings and experimental verifications with new catalysts. Because of the embedding of local environment, the model allows the atomic level of interpretation of the steric and electronic influence on the overall synthetic performance, which serves as a useful guide for the molecular engineering towards the target synthetic function. This model offers an extrapolative and interpretable approach for reaction performance prediction, pointing out the importance of chemical knowledge-constrained reaction modelling for synthetic purpose. 

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
