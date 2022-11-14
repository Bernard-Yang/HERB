# HERB


This repository contains the code for the AACL 2022 paper ["HERB: Measuring Hierarchical Regional Bias in Pre-trained Language Models"](https://arxiv.org/abs/2211.02882). Please cite the paper if you find it useful. 

This paper bridges the gap by analysing the regional bias learned by the pre-trained language models that are broadly used in NLP tasks. In addition to verifying the existence of regional bias in LMs, we find that the biases on regional groups can be strongly influenced by the geographical clustering of the groups. We accordingly propose a HiErarchical Regional Bias evaluation method (HERB) utilising the information from the sub-region clusters to quantify the bias in pre-trained LMs.

<div align=center><img width="543" alt="image" src="https://user-images.githubusercontent.com/45395508/200992600-97b23416-c211-451c-bdba-0223962c1da6.png">
  

Figure 1: The Regional Likelihood in [bald] Dimen- sion Produced by RoBERTa.

<div align=left>Run measureBias.sh for measuring the bias score in Table 1.
  
  
Replacing the file calculateBiasMeasure.py in measureBias.sh with calculateBiasVariant.py for the bias score in Table 2.
  
Run ablationDesTopics.py for Ablation study in Table 3.
  
Run measureBiasAbla.sh for Robustness Study in Table 6.  
