# SEMG-MIGNN
This is a repository for paper "Extrapolative and interpretable reaction performance predic-tion via chemical knowledge-based graph model".

# Introduction

The chemical comprehension and accurate prediction of reactivity and selectivity provide the foundation for the rational and efficient exploration of massive synthetic space. This establishment of the structure-performance relationship (SPR) has been focused on the reaction mechanism study and elucidation of the determining transition state model. Based on the transition state model, chemists can elucidate the origins of the observed reactivity/selectivity trend and make synthetic judgments based on chemical theory and empirical experience. This classic knowledge-driven strategy has reached remarkable success in synthetic chemistry and guided the discovery of hundreds of named reactions. Despite the advantage of offering qualitative guidance in the synthetic universe, it is challenging for the knowledge-driven strategy to handle the high-dimensional SPR without a clear mechanistic basis and analytic equation. The seemingly subtle change in catalyst, additive, or even solvent may result in significant perturbation of the overall synthetic performance. This is why laborious and repetitive condition optimization is still inevitably required, limiting the efficiency for synthetic development.
The data-driven approach has recently emerged as a powerful strategy for SPR establishment. By harnessing the interrelationship within the synthetic data, modern machine learning (ML) algorithm can create powerful models for synthetic prediction. Accurate predictions of reaction yield, chemo-, regio- and stereoselectivity have been achieved in a wide array of organic transformations, which validated the exciting concept of ML prediction of synthetic performances. However, the ML prediction and design of synthetic transformation are still far from mature; one of the major bottlenecks is the availability of the molecular encoding approach and the ML framework that are suitable for SPR prediction (Fig. 1a). Quantum chemical descriptors are known for the solid physical basis and high descriptive ability, but their generation is time- and resource-consuming, which hinders the application in high-throughput virtual screening. The string- and topological structure-based encodings (i.e. SMILES molecular fingerprints, etc.) do not require expert knowledge of the studied transformation and can be efficiently generated, while their chemical interpretability is low. In addition, the extrapolation problem presents additional challenges for SPR prediction. The needed guidance for new catalysts and transformations is still difficult for current synthetic models.

Unlike the current situation of synthetic performance prediction where most studies directly applied available ML algorithms, the innovation of ML framework for molecular property prediction has recently saw remarkable progress. Based on graph representation, a series of breakthroughs of molecular graph models have enabled chemical accuracy-level predictions for millions of small organic molecules (Fig. 1b), which has generated a strong momentum for artificial intelligence design of functional molecules. These studies revealed the potential of artificial intelligence modelling in chemical research and particularly pointed out the critical role of model framework in chemical ML. In light of the advantage of graph model in representing small organic molecules, we surmise that the SPR model can be innovated by enriching the local encodings of chemical environment and strengthen the information interaction between reaction components. Herein we report a new reaction performance model with two innovative designs; this model for the first time embeds the digitalized steric and electronic information of atomic environment, and the molecular interaction module allows the effective learning of the synergistic control by multiple reaction components (Fig. 1c). This model achieved excellent yield and stereoselectivity predictions, and our additional experimental tests of asymmetric thiol addition of imines verified its extrapolation ability in new catalyst predictions. This efficient, accurate and explainable model provides a useful approach for reaction performance prediction, which will accelerate the ML design of molecular synthesis.



# Packages requirements
In order to run Jupyter Notebook involved in this repository, several third-party python packages are required. The versions of these packages in our station are listed below.
```
matplotlib = 3.4.2
numpy = 1.22.4  
pandas = 1.3.3 
rdkit = 2022.03.2   
scipy = 1.4.1 
seaborn = 0.11.1 
sklearn = 0.23.2  
xgboost = 1.3.3 
```

# Demo & Instructions for use
Notebook 1 demonstrates how to use demo data to 
# How to cite
The paper is under review.
# Contact with us
Email: hxchem@zju.edu.cn; 22037023@zju.edu.cn
