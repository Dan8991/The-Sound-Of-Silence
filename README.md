#  The Sound of Silence: Efficiency of First Digit Features in Synthetic Audio Detection

This repository contains the code used to develop the algorithm described in the paper "The Sound of Silence: Efficiency of First Digit Features in Synthetic Audio Detection" (Accepted at WIFS 2022) by [Daniele Mari](https://github.com/Dan8991), [Federica Latora](https://github.com/FedericaLatora) and [Simone Milani](https://github.com/profmilani).

## Installation
### Dataset
First of all download the ASVSpoof dataset from https://www.asvspoof.org/database (we used the 2019 version)
```
mkdir datasets
mkdir datasets/raw
mkdir datasets/processed
mkdir datasets/raw/ASVspoof-LA
mkdir datasets/processed/ASVspoof-LA
```
Then move the content of the LA folder from ASVSpoof inside datasets/raw/ASVspoof2019_LA.
### Environment
```
conda env create -f environment.yml
```


### Preprocessing
From the root folder run

```
python src/data_splitting.py
```

to create train, dev and eval datasets. Then execute

```
python src/feature_extraction.py --type {type}
```

to generate the features. You can use argument --type to choose if you want to generate the full features, only the once for silenced parts or only the voiced parts.

### Training 

Run
```
python src/rf.py --type {type} --base {base} --q {q}
```
to train the random forest classifiers and to get the results. The type argument is used to select the type of features, the base argument to choose the benford law base and the q argument to choose the scaling factors.
