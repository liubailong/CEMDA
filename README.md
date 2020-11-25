# Combined Embedding Model for MiRNA-Disease Association Prediction（CEMDA）


### Implemented evironment
Python>=3.6


###Required libraries
`numpy,numba,openpyxl,xlrd，torch,itertools,sys,os,importlib`

We recommended that you could install Anaconda to meet these requirements


### How to run CEMDA? 
####Data
All datas or mid results are orgnized in `DATA` fold, which contains miRNA-disease associations,disease semantic similarity, miRNA functional similarity, encode result of disease and  miRNA.

####The starting point for running CEMDA is:

(1)**meta_path_instance.py**：gereating meta-paths from the dataset of miRNA-disease associations,disease semantic similarity, miRNA functional similarity. all the result is saved in the folds named `"5.mid result"` and `"6.meta path"`, which need to be created by yourselves.

(2)**CEMDA.py**: training the model of PESLMDA which will referece `GRU.py, MLP.py,SelfAttention.py. `
And it outputs the parameter of CEMDA

####other relateive files:
**GRU.py**: a GRU model in CEMDA
**MLP.py**: a MLP model in CEMDA
**SelfAttention.py**:SelfAttention model in CEMDA
