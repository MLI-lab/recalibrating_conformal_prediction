from pathlib import Path
import json
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import itertools

from typing import Callable, Dict, Union

from robustness.tools import breeds_helpers
import robustness.datasets

import sys
sys.path.append(str(Path(__name__).resolve().parent.parent))

# data utils
from utils.pytorch_utils.data_utils import get_dataset
from utils.pytorch_utils.model_utils import fetch_checkpoints_by_params, weights_update
from utils.pytorch_utils.inference import get_sorted_softmax, get_logits, confidence_histogram_with_accuracy, get_class_confidence_accuracy
from utils.pytorch_utils.prediction import MLPConformalPredictor, QuantileConfidenceThreshold

# conformal utils
from utils.conformal_utils.conformal import ConformalModel, ThresholdedConformalModel


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


r"""All prediction methods and their configurations"""
# methods and configurations
regression_methods_configs = [
    {
        'method': 'confidence_histogram_regression',
        'model_args': {
            'num_features': 10,
            'feature_extractor': lambda logits: confidence_histogram_with_accuracy(logits, bins=10)[0] / len(logits)
        }
    },
    {
        'method': 'confidence_histogram_less_last_bin_regression',
        'model_args': {
            'num_features': 9,
            'feature_extractor': lambda logits: confidence_histogram_with_accuracy(logits, bins=10)[0][:-1]  / len(logits)
        }
    },
    {
        'method': 'predicted_class_confidence_regression',
        'model_args': {
            'num_features': 1000,
            'feature_extractor': lambda logits: get_class_confidence_accuracy(logits, num_classes=1000, by_predictions=True, replace_nans=True)[0]
        }
    },
    {
        'method': 'average_confidence_regression',
        'model_args': {
            'num_features': 1,
            'feature_extractor': lambda logits: get_sorted_softmax(logits.tensors[0])[0][:,0].mean()
        }
    },
    {
        'method': 'difference_of_confidence_regression',
        'model_args': {
            'num_features': 1,
            'feature_extractor': lambda logits, offset: get_sorted_softmax(logits.tensors[0])[0][:,0].mean() - offset
        }
    }
]


# Conformal prediction class mapping
conformal_model_constructor_dict: Dict[str, Callable] = {
    'TPS': ThresholdedConformalModel,
    'APS': ConformalModel,
    'RAPS': ConformalModel
}




if __name__ == "__main__":

    # same as the filename
    cache_fname = './cache/breeds_tps_results.csv'

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
    datasets = ['ImageNetV2', 'ImageNet-R', 'ImageNet-Sketch']

    # whether tau or alpha is predicted
    target_param = 'tau'

    # dataset hparams
    # use_classes_of_datasets = [None, 'ImageNet-R']
    use_classes_of_datasets = [None]
    breeds_datasets = ['entity13', 'entity30', 'living17', 'nonliving26']
    breeds_split = 'rand'

    # pre-trained classifier
    arches = ['resnet18']

    # checkpoints
    classifier_checkpoint_dir = "../ckpts"
    regression_checkpoint_dir = Path('./outputs')

    # data dir
    info_dir = "../../clone_repos/BREEDS-Benchmarks/imagenet_class_hierarchy/modified"
    data_root_dir = "../../imagenet-testbed/s3_cache/datasets"
    
    # dataset utils params
    use_encodings = False

    # data params
    batch_size = 128
    num_workers = 4

    # gpu
    gpu = 0


    """conformal parameters"""
    alphas = [0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.01]
    # alphas = [0.1]

    randomized = True
    allow_zero_sets = True

    # platt scaling
    # platt_scaling_uses = [False, True]
    platt_scaling_uses = [False]


    conformal_params = []
    
    # APS
    aps_kregs = [-1]
    aps_lamdas = [0]

    conformal_params += list(itertools.product(['APS'], alphas, aps_kregs, aps_lamdas, platt_scaling_uses))
    
    # RAPS
    raps_kregs = [2, 4]
    raps_lamdas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    conformal_params += list(itertools.product(['RAPS'], alphas, raps_kregs, raps_lamdas, platt_scaling_uses))

    # TPS
    tps_kregs = [-1]
    tps_lamdas = [0]

    conformal_params += list(itertools.product(['TPS'], alphas, tps_kregs, tps_lamdas, platt_scaling_uses))
    
    

    try:
        df = pd.read_csv(cache_fname)
        print("[INFO] Using cache file for the conformal thresholds...")

    except FileNotFoundError as e:
        ### Perform the experiment
        df = pd.DataFrame(columns = ["Dataset","Method","Classifier","ConformalPredictor","Top1","Top5","alpha","raps_kreg","raps_lambda","use_platt_scaling","calibrated_tau","original_coverage","original_size","oracle_tau","predicted_alpha","predicted_tau","predicted_coverage","predicted_size"])

        for arch, breeds_dataset, use_classes_of_dataset in itertools.product(arches, breeds_datasets, use_classes_of_datasets):

            # get breeds pretrained checkpoint
            params_search_dict = {'arch': arch, 'breeds_dataset': breeds_dataset}
            breeds_checkpoints = fetch_checkpoints_by_params(classifier_checkpoint_dir, hparams_filename='hparams.yaml', checkpoint_filename_ext='ckpt', verbose=False, search_dict=params_search_dict)

            if len(breeds_checkpoints) != 1:
                raise RuntimeError(f"Number of pretrained checkpoints found for this config is expected to be 1, but got {len(breeds_checkpoints)}")
            
            # Get the model
            if use_encodings:
                model = models.__dict__[arch](pretrained=True).fc
            else:
                model = models.__dict__[arch](pretrained=True)

            num_classes = int(breeds_dataset[-2:])

            # Replace the linear layer. 
            # WARNING: Only works for Resnets. 
            # TODO: other arches.
            num_filters = model.fc.in_features
            model.fc = nn.Linear(num_filters, num_classes)

            # load pretrained checkpoint
            # model.load_state_dict(torch.load(breeds_checkpoints[0]))
            weights_update(model, torch.load(breeds_checkpoints[0]))

            if gpu is not None:
                model = model.cuda(gpu)

            
            breeds_constructor = getattr(breeds_helpers, 'make_' + breeds_dataset)

            split_metadata = breeds_constructor(info_dir, split=breeds_split)
            train_subclasses, test_subclasses = split_metadata[1]

            dataset = robustness.datasets.CustomImageNet(data_root_dir, train_subclasses)

            _, val_loader = dataset.make_loaders(num_workers, batch_size, only_val=True)

            val_logits = get_logits(
                model,
                val_loader,
                is_data_loader=True,
                num_classes=num_classes,
                batch_size=batch_size,
                num_workers=num_workers,
                gpu=gpu
            )


            for cur_dataset_name in tqdm([breeds_dataset], desc='Iter. datasets'):

                dataset = robustness.datasets.CustomImageNet(data_root_dir, test_subclasses)

                _, cur_loader = dataset.make_loaders(num_workers, batch_size, only_val=True)

                cur_logits = get_logits(
                    model,
                    cur_loader,
                    is_data_loader=True,
                    num_classes=num_classes,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    gpu=gpu
                )


                for conf_predictor, alpha, kreg, lamda, use_platt_scaling in conformal_params:
                    
                    conformal_class_constructor = conformal_model_constructor_dict[conf_predictor]

                    conformal_model = conformal_class_constructor(
                        model,
                        val_logits,
                        is_logits = True,
                        alpha = alpha, 
                        randomized = randomized,
                        allow_zero_sets = allow_zero_sets,
                        use_platt_scaling = use_platt_scaling,
                        kreg = kreg,
                        lamda = lamda,
                        batch_size = batch_size,
                        num_workers = num_workers,
                        gpu = gpu
                    )
                    oracle_model = conformal_class_constructor(
                        model,
                        cur_logits,
                        is_logits = True,
                        alpha = alpha, 
                        randomized = randomized,
                        allow_zero_sets = allow_zero_sets,
                        use_platt_scaling = use_platt_scaling,
                        kreg = kreg,
                        lamda = lamda,
                        batch_size = batch_size,
                        num_workers = num_workers,
                        gpu = gpu
                    )

                    orig_calib_tau = conformal_model.q_hat
                    oracle_tau = oracle_model.q_hat

                    softmax_temp = conformal_model.softmax_temp

                    orig_top1, orig_top5, orig_cvg, orig_sz = conformal_model.validate_conformal(cur_logits, is_logits=True)

                    r"""search regression methods with existing checkpoint"""
                    hparam_search_dict = {'classifier': arch, 'breeds_dataset': cur_dataset_name, 'alpha': alpha}
                    exclude_dict = {'qtc_boosted': True}

                    hparam_search_dict.update({'conformal_predictor': conf_predictor})
                    if use_classes_of_dataset:
                        hparam_search_dict.update({'limit_classes_by_dataset': use_classes_of_dataset.lower()})
                    if use_platt_scaling is True:
                        hparam_search_dict.update({'use_platt_scaling': use_platt_scaling})
                    if lamda != 0:
                        hparam_search_dict.update({'raps_kreg': kreg, 'raps_lambda': lamda})

                    if target_param != 'tau':
                        hparam_search_dict.update({'target_param': target_param})
                    else:
                        exclude_dict.update({'target_param': 'alpha'})
                    matching_regression_configs = []
                    for cur_method_config in regression_methods_configs:
                        match_checkpoints = fetch_checkpoints_by_params(
                            regression_checkpoint_dir / cur_method_config['method'],
                            verbose=True,
                            search_dict=hparam_search_dict,
                            exclude_dict=exclude_dict
                        )
                        if match_checkpoints:
                            match_config = {k: v for k,v in cur_method_config.items()}
                            match_config.update({'checkpoint_file': match_checkpoints[-1]})
                            matching_regression_configs.append(match_config)
           
                    if not matching_regression_configs:
                        print(f'[INFO] No matching regression method checkpoints were provided for\t---\t classifier: {arch} | alpha: {alpha}')

                    for cur_method_config in matching_regression_configs:

                        # set feature extractor of DoC method
                        if cur_method_config['method'] == "difference_of_confidence_regression":
                            offset = get_sorted_softmax(val_logits.tensors[0])[0][:,0].mean()
                            cur_method_config['model_args']['feature_extractor']= lambda logits: get_sorted_softmax(logits.tensors[0])[0][:,0].mean() - offset

                        mlp_model = MLPConformalPredictor(**cur_method_config['model_args'], gpu=gpu)
                        mlp_model.load_from_checkpoint(checkpoint_file=cur_method_config['checkpoint_file'])

                        predicted_tau = mlp_model.predict_by_logits(cur_logits)

                        # account for the offset of DoC method
                        if cur_method_config['method'] == "difference_of_confidence_regression":
                            predicted_tau += orig_calib_tau

                        # set tau of conformal model to predicted tau in order to get coverage results
                        conformal_model.q_hat = predicted_tau

                        cur_top1, cur_top5, cur_cvg, cur_sz = conformal_model.validate_conformal(cur_logits, is_logits=True)


                        df = df.append({"Dataset": cur_dataset_name,
                                        "on_classes_of": use_classes_of_dataset,
                                        "Method": cur_method_config['method'],
                                        "Classifier": arch,
                                        "ConformalPredictor": conf_predictor,
                                        "Top1": np.round(orig_top1,3),
                                        "Top5": np.round(orig_top5,3),
                                        "alpha": alpha,
                                        "raps_kreg": kreg,
                                        "raps_lambda": lamda,
                                        "use_platt_scaling": use_platt_scaling,
                                        "calibrated_tau": orig_calib_tau,
                                        "original_coverage": np.round(orig_cvg,3),
                                        "original_size": np.round(orig_sz,3),
                                        "oracle_tau": oracle_tau,
                                        "predicted_tau": predicted_tau,
                                        "predicted_coverage": np.round(cur_cvg,3),
                                        "predicted_size": np.round(cur_sz,3)}, ignore_index=True)


                        # save the current progress
                        df.to_csv(cache_fname)


                    ####################   QTC   ####################
                    cur_qtc = QuantileConfidenceThreshold(cur_logits.tensors[0], cur_logits.tensors[1], quantile=alpha, softmax_temperature=softmax_temp)

                    predicted_alpha = cur_qtc.predict(val_logits.tensors[0])

                    # recalibrate conformal model with predicted 1-alpha
                    conformal_model = conformal_class_constructor(
                        model,
                        val_logits,
                        is_logits = True,
                        alpha = predicted_alpha, 
                        randomized = randomized,
                        allow_zero_sets = allow_zero_sets,
                        use_platt_scaling = softmax_temp,
                        kreg = kreg,
                        lamda = lamda,
                        batch_size = batch_size,
                        num_workers = num_workers,
                        gpu = gpu
                    )

                    # get predicted_tau from the recalibrated conformal model with the predicted alpha
                    predicted_tau = conformal_model.q_hat

                    cur_top1, cur_top5, cur_cvg, cur_sz = conformal_model.validate_conformal(cur_logits, is_logits=True)

                    df = df.append({"Dataset": cur_dataset_name,
                                    "on_classes_of": use_classes_of_dataset,
                                    "Method": 'QTC',
                                    "Classifier": arch,
                                    "ConformalPredictor": conf_predictor,
                                    "Top1": np.round(orig_top1,3),
                                    "Top5": np.round(orig_top5,3),
                                    "alpha": alpha,
                                    "raps_kreg": kreg,
                                    "raps_lambda": lamda,
                                    "use_platt_scaling": use_platt_scaling,
                                    "calibrated_tau": orig_calib_tau,
                                    "original_coverage": np.round(orig_cvg,3),
                                    "original_size": np.round(orig_sz,3),
                                    "oracle_tau": oracle_tau,
                                    "predicted_alpha": predicted_alpha,
                                    "predicted_tau": predicted_tau,
                                    "predicted_coverage": np.round(cur_cvg,3),
                                    "predicted_size": np.round(cur_sz,3)}, ignore_index=True)


                    
                    ####################   QTC-SC   ####################
                    cur_qtc = QuantileConfidenceThreshold(val_logits.tensors[0], val_logits.tensors[1], quantile=1.0-alpha, softmax_temperature=softmax_temp)

                    predicted_alpha = 1.0 - cur_qtc.predict(cur_logits.tensors[0])

                    # recalibrate conformal model with predicted 1-alpha
                    conformal_model = conformal_class_constructor(
                        model,
                        val_logits,
                        is_logits = True,
                        alpha = predicted_alpha, 
                        randomized = randomized,
                        allow_zero_sets = allow_zero_sets,
                        use_platt_scaling = softmax_temp,
                        kreg = kreg,
                        lamda = lamda,
                        batch_size = batch_size,
                        num_workers = num_workers,
                        gpu = gpu
                    )

                    # get predicted_tau from the recalibrated conformal model with the predicted alpha
                    predicted_tau = conformal_model.q_hat

                    cur_top1, cur_top5, cur_cvg, cur_sz = conformal_model.validate_conformal(cur_logits, is_logits=True)

                    df = df.append({"Dataset": cur_dataset_name,
                                    "on_classes_of": use_classes_of_dataset,
                                    "Method": 'QTC-SC',
                                    "Classifier": arch,
                                    "ConformalPredictor": conf_predictor,
                                    "Top1": np.round(orig_top1,3),
                                    "Top5": np.round(orig_top5,3),
                                    "alpha": alpha,
                                    "raps_kreg": kreg,
                                    "raps_lambda": lamda,
                                    "use_platt_scaling": use_platt_scaling,
                                    "calibrated_tau": orig_calib_tau,
                                    "original_coverage": np.round(orig_cvg,3),
                                    "original_size": np.round(orig_sz,3),
                                    "oracle_tau": oracle_tau,
                                    "predicted_alpha": predicted_alpha,
                                    "predicted_tau": predicted_tau,
                                    "predicted_coverage": np.round(cur_cvg,3),
                                    "predicted_size": np.round(cur_sz,3)}, ignore_index=True)
                    
                    
                    ####################   QTC-ST   ####################
                    cur_qtc = QuantileConfidenceThreshold(val_logits.tensors[0], val_logits.tensors[1], quantile=orig_calib_tau, softmax_temperature=softmax_temp)

                    predicted_tau = cur_qtc.predict(cur_logits.tensors[0])

                    # set tau of conformal model to predicted tau in order to get coverage results
                    conformal_model.q_hat = predicted_tau

                    cur_top1, cur_top5, cur_cvg, cur_sz = conformal_model.validate_conformal(cur_logits, is_logits=True)

                    df = df.append({"Dataset": cur_dataset_name,
                                    "on_classes_of": use_classes_of_dataset,
                                    "Method": 'QTC-ST',
                                    "Classifier": arch,
                                    "ConformalPredictor": conf_predictor,
                                    "Top1": np.round(orig_top1,3),
                                    "Top5": np.round(orig_top5,3),
                                    "alpha": alpha,
                                    "raps_kreg": kreg,
                                    "raps_lambda": lamda,
                                    "use_platt_scaling": use_platt_scaling,
                                    "calibrated_tau": orig_calib_tau,
                                    "original_coverage": np.round(orig_cvg,3),
                                    "original_size": np.round(orig_sz,3),
                                    "oracle_tau": oracle_tau,
                                    "predicted_tau": predicted_tau,
                                    "predicted_coverage": np.round(cur_cvg,3),
                                    "predicted_size": np.round(cur_sz,3)}, ignore_index=True)


                    df.to_csv(cache_fname)





