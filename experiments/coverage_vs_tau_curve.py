from pathlib import Path
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__name__).resolve().parent.parent))

# data utils
from utils.pytorch_utils.data_utils import get_dataset
from utils.pytorch_utils.inference import get_logits

# conformal utils
from utils.conformal_utils.conformal import ConformalModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models



    


if __name__ == "__main__":

    # same as the filename
    cache_fname = './cache/coverage_vs_tau.csv'

    # random seed
    seed = 42

    # Set random states
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

    cudnn.benchmark = True

    # dataset paths
    val_dataset_name = 'ImageNet-Val'
    shift_dataset_name = 'ImageNet-Sketch'

    # pre-trained classifier
    arch = 'resnet50'
    
    # dataset utils params
    use_encodings = False

    # data params
    batch_size = 128
    num_workers = 4

    # gpu
    gpu = 0


    """conformal parameters"""
    alpha = 0.1
    num_tau_samples = 100
    tau_min, tau_max = 0.75, 1.0
    taus = np.linspace(tau_min, tau_max, num_tau_samples)

    randomized = True
    allow_zero_sets = True

    # APS
    kreg = -1
    lamda = 0


    try:
        df = pd.read_csv(cache_fname)
        print("[INFO] Using cache file for the conformal thresholds...")

    except FileNotFoundError as e:
        ### Perform the experiment
        df = pd.DataFrame(columns = ["Dataset","Classifier","ConformalPredictor","alpha","tau","coverage","size"])

        # get pretrained model
        if use_encodings:
            model = models.__dict__[arch](pretrained=True).fc
        else:
            model = models.__dict__[arch](pretrained=True)

        if gpu is not None:
            model = model.cuda(gpu)

        
        val_dataset = get_dataset(val_dataset_name, dataset_type='raw')
        shift_dataset = get_dataset(shift_dataset_name, dataset_type='raw')

        val_logits = get_logits(
            model,
            val_dataset,
            batch_size=batch_size, 
            num_workers=num_workers, 
            gpu=gpu
        )
        shift_logits = get_logits(
            model,
            shift_dataset,
            batch_size=batch_size, 
            num_workers=num_workers, 
            gpu=gpu
        )


        conformal_model = ConformalModel(
            model,
            val_logits,
            is_logits = True,
            alpha = alpha, 
            randomized = randomized,
            allow_zero_sets = allow_zero_sets,
            kreg = kreg,
            lamda = lamda,
            batch_size = batch_size,
            num_workers = num_workers,
            gpu = gpu
        )


        for cur_tau in tqdm(taus, desc=f'tau from {tau_min} to {tau_max}'):

            conformal_model.q_hat = cur_tau

            val_top1, val_top5, val_cvg, val_sz = conformal_model.validate_conformal(val_logits, is_logits=True)
            shift_top1, shift_top5, shift_cvg, shift_sz = conformal_model.validate_conformal(shift_logits, is_logits=True)


            df = df.append({"Dataset": val_dataset_name,
                            "Classifier": arch,
                            "ConformalPredictor": "APS",
                            "alpha": alpha,
                            "tau": cur_tau,
                            "coverage": np.round(val_cvg,3),
                            "size": np.round(val_sz,3)}, ignore_index=True)
            df = df.append({"Dataset": shift_dataset_name,
                            "Classifier": arch,
                            "ConformalPredictor": "APS",
                            "alpha": alpha,
                            "tau": cur_tau,
                            "coverage": np.round(shift_cvg,3),
                            "size": np.round(shift_sz,3)}, ignore_index=True)
  
            # save the current progress
            df.to_csv(cache_fname)



