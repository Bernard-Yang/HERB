# HERB


This repository contains the code for the AACL 2022 paper "HERB: Measuring Hierarchical Regional Bias in Pre-trained Language Models". Please cite the paper if you find it useful. 

This paper bridges the gap by analysing the regional bias learned by the pre-trained language models that are broadly used in NLP tasks. In addition to verifying the existence of regional bias in LMs, we find that the biases on regional groups can be strongly influenced by the geographical clustering of the groups. We accordingly propose a HiErarchical Regional Bias evaluation method (HERB) utilising the information from the sub-region clusters to quantify the bias in pre-trained LMs.

<img width="543" alt="image" src="https://user-images.githubusercontent.com/45395508/200992600-97b23416-c211-451c-bdba-0223962c1da6.png", div align=center>
  

Figure 1: The Regional Likelihood in [bald] Dimen- sion Produced by RoBERTa.

  
Run measureBias.sh for measuring the bias score of Table1 in paper. 
