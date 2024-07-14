# Sedihist: Grain Size Analysis of Images from Sedimentary Simulation Experiments

## About Sedihist
We proposed a deep learning model for analyzing images from  Sedimentary Simulation Experiments with fuzzy grain edges and irregular arrangements.

This repository contains code and examples 

Sedihist can be configured and trained to estimated the grain size corresponding to the nine cumulative volume percentage from a single input image.

You can use the models in this repository. or you can train Sedihist for your own purpose on you datasets.
## Instrallation Prerequisites
This code uses python, pytorch, and barbar.Barbar is used to show the progress of model.

## Main Functions
1. Intialize model
```model, input_size = intialize_model(**Parameters)```
2. Train model
```train_dict = train_model(**Parameters)```
3. Test model
```test_dict = test_model(**Parameters)```

### Train Sedihist
Run ```train_Sedihist.py```
### Test Sedihist
Run ```test_Sedihist.py```
### paremeters setting
Modify input data address in ```Prepare_Data.py``` 
Modify output result address in ```Network_functions.py ```
Modify model saving address in ``` Network_functions.py``` 
Modify window size and bin numbers in ```Demo_Parameters.py```


