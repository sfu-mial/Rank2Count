# Rank2Count
The official repository for the paper "Learning-to-Count by Learning-to-Rank", accepted as an oral presentation at CRV2023. We attempt to solve the problem of weakly supervised object counting using pairwise image ranking.

## Motivation
Fully supervised object counting methods typically rely on density map annotations, which are labor intensive to collect. We propose a novel method to exploit pairwise image ranking, which is a significantly weaker form of annotations. These annotations require an annotator to estimate a boolean label $r_{ij} = c_i > c_j$ for an image pair $(x_i, x_j)$, where $c_i$ and $c_j$ represent the true but unknown object count for each image.  
In addition to learning directly from pairwise image annotations, we introduces a novel adversarial regularization loss, which encourages the network output to have the structure of a density map while also solving the pairwise ranking problem.
<p align="center">
  <img src="/figures/method.png" height=200px />
</p>

## Repository layout
```
|- src 
|- scripts
|- train.py
```

## Installation
### Code

### Datasets
Penguins: https://www.robots.ox.ac.uk/~vgg/data/penguins/  
Trancos: https://gram.web.uah.es/data/datasets/trancos/index.html  
Mall: https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html

#### Data format



