# Sedihist: Grain Size Analysis of Images from Sedimentary Simulation Experiments

## About Sedihist
We proposed a deep learning model for analyzing images from  Sedimentary Simulation Experiments with fuzzy grain edges and irregular arrangements.

This repository contains code and examples 

Sedihist can be configured and trained to estimated the grain size corresponding to the nine cumulative volume percentage from a single input image.

You can use the models in this repository. or you can train Sedihist for your own purpose on you datasets.
## Instrallation Prerequisites
This code uses python, pytorch
## Main Functions
1. Intialize model
```model, input_size = intialize_model(**Parameters)```
2. Train model
```train_dict = train_model(**Parameters)```
3. Test model
```test_dict = test_model(**Parameters)```