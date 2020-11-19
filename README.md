# Predicting MiRNA-disease Associations by Pair Embedding and Sequence Learning Model（PESLMDA）


### Implemented evironment
Python>=3.6


###Required libraries
`numpy,numba,openpyxl,xlrd，torch,itertools,sys,os,importlib`

We recommended that you could install Anaconda to meet these requirements


### How to run PESLMDA? 
####Data
All datas or mid results are orgnized in `DATA` fold, which contains miRNA-disease associations,disease semantic similarity, miRNA functional similarity, encode result of disease and  miRNA.

####The starting point for running PESLMDA is:

(1)**meta_path_instance.py**：gereating meta-paths from the dataset of miRNA-disease associations,disease semantic similarity, miRNA functional similarity. all the result is saved in the folds named `"5.mid result"` and `"6.meta path"`, which need to be created by yourselves.

(2)**PESLMDA.py**: training the model of PESLMDA which will referece `GRU.py, MLP.py,MLP.py,SelfAttention.py. `
And it outputs the parameter of PESLMDA

####other relateive files:
**GRU.py**: a GRU model in PESLMDA
**MLP.py**: a MLP model in PESLMDA
**SelfAttention.py**:SelfAttention model in PESLMDA