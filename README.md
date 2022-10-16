# Recalibrating Conformal Prediction

Test-time recalibration of conformal predictors based on unlabeled data for improved performance under test distribution shift.

# Requirements
The following Python libraries are required to run the code in this repository:

```
torch
torchvision
```
Other requirements can be installed with `pip install -r requirements.txt`. The above two libraries are not included in `requirements.txt` for safe measure.

# Usage
All the figures in the paper can be reproduced by [running this notebook](./notebooks/recalibration_results.ipynb)

## Citation
```
@article{yilmazheckel2022RecalibratingConformal,
    author    = {Fatih Furkan Yilmaz and Reinhard Heckel},
    title     = {Test-time Recalibration of Conformal Predictors Under Distribution Shift Based on Unlabeled Examples},
    journal   = {arXiv:2210.04166},
    year      = {2022}
}
```

## Licence
All files are provided under the terms of the Apache License, Version 2.0.