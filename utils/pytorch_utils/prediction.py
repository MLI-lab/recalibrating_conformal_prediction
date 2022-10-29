from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Union

# data utils
from .data_utils import get_dataset
from .inference import get_logits, get_sorted_softmax
from .regression import mlp, fit_regressor

import torch



class QuantileConfidenceThreshold:

    score_choices = {
        'confidence': ['confidence', 'conf', 'softmax'],
        'entropy': ['entropy', 'negative_entropy']
    }

    def __init__(self, logits: torch.Tensor, targets: torch.Tensor, score: str = 'confidence', quantile: Union[float, None] = None, jitter: Union[float, None] = None, softmax_temperature: Union[float, torch.Tensor, None] = None) -> None:
        r"""
        Implements the Quantile Confidence Threhold (QTC) method.

        Args:
            logits (torch.Tensor): raw model outputs with dimensions (num_examples, num_classes).
            targets (torch.Tensor): one dimensional tensor containing correct labels
            score (str, optional): score to use for examples. choices: confidence, negative entropy (default: ``confidence``). 
            quantile (float): which quantile to take of the array of scores.
            jitter (float): small perturbation applied to the scores. This might help if the scores are too close to one another or to math:`1.0`.
            softmax_temperature (float): Temperature scaling used before the softmax mapping.
        """

        if quantile is None:
            print(DeprecationWarning("Not specifying a quantile corresponds to ATC method, which is deprecated."))
        
        self.score_func = self._get_score(score)
        self.jitter = jitter
        self.softmax_temp = softmax_temperature
        self.q = self._get_threshold(logits, targets, quantile=quantile)


    def predict(self, logits: torch.Tensor) -> float:

        softmax_scores, _ = get_sorted_softmax(logits, temperature=self.softmax_temp)

        if self.score_func == 'confidence':
            sorted_scores, _ = softmax_scores[:,0].sort()
        else:
            neg_entropy = (softmax_scores * softmax_scores.log()).sum(dim=1)
            sorted_scores, _ = neg_entropy.sort()

        # self.jitter is supposed to help if there are too many examples with the same confidence (usually 1.0)
        if self.jitter:
            jitter_arr = torch.arange(sorted_scores.numel()-1, -1, -1)
            sorted_scores -= self.jitter * jitter_arr

        pred = (sorted_scores < self.q).sum() / sorted_scores.numel()

        return pred.item()

    
    def predict_accuracy(self, logits: torch.Tensor) -> float:
        err1_pred = self.predict(logits)
        acc1_pred = 1.0 - err1_pred
        return acc1_pred

    
    def _get_threshold(self, logits: torch.Tensor, targets: torch.Tensor, quantile: Union[float, None] = None) -> torch.Tensor:

        if quantile:
            # assert self.score_func == 'confidence'
            assert (quantile > 0.0) and (quantile < 1.0)
            err1 = torch.Tensor([quantile])
        else:
            acc1 = (logits.argmax(dim=1) == targets).sum() / targets.size(0)
            err1 = torch.Tensor([1.0]) - acc1.item()

        softmax_scores, _ = get_sorted_softmax(logits, temperature=self.softmax_temp)

        if self.score_func == 'confidence':
            sorted_scores, _ = softmax_scores[:,0].sort()
        else:
            neg_entropy = (softmax_scores * softmax_scores.log()).sum(dim=1)
            sorted_scores, _ = neg_entropy.sort()

        # self.jitter is supposed to help if there are too many examples with the same confidence (usually 1.0)
        if self.jitter:
            jitter_arr = torch.arange(sorted_scores.numel()-1, -1, -1)
            sorted_scores -= self.jitter * jitter_arr

        thresh = sorted_scores.quantile(err1)

        return thresh

    def _get_score(self, score: str) -> str:
        if score in self.score_choices['confidence']:
            return 'confidence'
        elif score in self.score_choices['entropy']:
            return 'entropy'
        else:
            raise ValueError(f"Invalid score setting for {score}")



class AverageConfidenceThreshold:

    score_choices = {
        'confidence': ['confidence', 'conf', 'softmax'],
        'entropy': ['entropy', 'negative_entropy']
    }

    def __init__(self, logits, targets, score='confidence', tau_quantile=None) -> None:

        if tau_quantile is not None:
            print(DeprecationWarning("This class is deprecated for the use of quantiles other than classification error. Please use `QuantileConfidenceThreshold` instead."))
        
        self.score_func = self._get_score(score)
        self.tau = self._get_tau(logits, targets, tau_quantile=tau_quantile)


    def predict_error(self, logits):

        softmax_scores, _ = get_sorted_softmax(logits)

        if self.score_func == 'confidence':
            sorted_scores, _ = softmax_scores[:,0].sort()
        else:
            neg_entropy = (softmax_scores * softmax_scores.log()).sum(dim=1)
            sorted_scores, _ = neg_entropy.sort()

        err1_pred = (sorted_scores < self.tau).sum() / sorted_scores.numel()

        return err1_pred.item()

    
    def predict_accuracy(self, logits):
        err1_pred = self.predict_error(logits)
        acc1_pred = 1.0 - err1_pred
        return acc1_pred

    
    def _get_tau(self, logits, targets, tau_quantile=None):

        if tau_quantile:
            assert self.score_func == 'confidence'
            assert (tau_quantile > 0.0) and (tau_quantile < 1.0001)
            err1 = torch.Tensor([tau_quantile])
        else:
            acc1 = (logits.argmax(dim=1) == targets).sum() / targets.size(0)
            err1 = torch.Tensor([1.0]) - acc1.item()

        softmax_scores, _ = get_sorted_softmax(logits)

        if self.score_func == 'confidence':
            sorted_scores, _ = softmax_scores[:,0].sort()
        else:
            neg_entropy = (softmax_scores * softmax_scores.log()).sum(dim=1)
            sorted_scores, _ = neg_entropy.sort()

        tau = sorted_scores.quantile(err1)

        return tau


    def _get_score(self, score):
        if score in self.score_choices['confidence']:
            return 'confidence'
        elif score in self.score_choices['entropy']:
            return 'entropy'
        else:
            raise ValueError(f"Invalid score setting for {score}")




class MLPConformalPredictor:

    def __init__(self, num_features: int = 1, **kwargs) -> None:

        # if num_features is not None:
        #     # initialize without model in case we fit a new model or 
        #     # load from checkpoint, which will be separate functions
        #     self.model = mlp(feature_dim=num_features)
        # else:
            
        self.model = mlp(feature_dim=num_features)
        self.num_features = num_features

        self.feature_extractor = None

        self.defaults_dict = {
            'model_args': {
                'lr': 1e-4, 
                'weight_decay': 1e-3, 
                'momentum': 0.9, 
                'max_iter': int(5e4), 
                'gpu': 0, 
                'outdir': None, 
                'freq': 20
            }
        }
        self._load_attributes_by_kwargs(kwargs)

        self.gpu = self.defaults_dict['model_args']['gpu']



    def predict_by_dataset(self, dataset, model, use_encodings: bool = False, arch=None, batch_size=128, num_workers=4, gpu=0, feature_extractor=None):
        if isinstance(dataset, str):
            # get dataset by title
            if use_encodings:
                assert arch, f"if use_encodings is set True, architecture must be specified, but {arch} was given."

            dataset = get_dataset(dataset, use_encodings=use_encodings, arch=arch)

        logits = get_logits(
            model,
            dataset,
            batch_size=batch_size, 
            num_workers=num_workers, 
            gpu=gpu
        )

        return self.predict_by_logits(logits.tensors[0], feature_extractor=feature_extractor)
    
    
    def predict_by_logits(self, logits, feature_extractor=None):
        if feature_extractor is None:
            assert self.feature_extractor, "Object feature_extractor and function argument extractor cannot be unspecified at the same time!"
            feature_extractor = self.feature_extractor

        features = feature_extractor(logits)

        return self.predict_by_features(features)

    
    def predict_by_features(self, input: torch.Tensor):
        if input.ndim < 2:
            input = input.unsqueeze(0)

        if self.gpu is not None:
            self.model = self.model.cuda(self.gpu)
            input = input.cuda(self.gpu)

        self.model.eval()
        with torch.no_grad():
            output = self.model(input)

        # return the value instead of tensor if scalar
        if output.numel() == 1:
            output = output.item()
            
        return output
    
    
    def fit_model(self, Xs, ys, **kwargs):
        params = {k: v for k, v in self.defaults_dict['model_args'].items()}
        params.update(kwargs)
        
        model, losses = fit_regressor(Xs, ys, **params)

        self.model = model
        self.losses = losses
    
    
    def load_from_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['state_dict'])


    def _set_feature_extractor(self, feature_extractor):
        if not callable(feature_extractor):
            raise TypeError(f"Feature extractor needs to be callable, but {type(feature_extractor)} type is given")
        
        self.feature_extractor = feature_extractor

    
    def _load_attributes_by_kwargs(self, args):
        # override model parameters
        cur_keywords = list(self.defaults_dict['model_args'].keys())
        self.defaults_dict['model_args'].update({k: v for k, v in args.items() if k in cur_keywords})

        # set feature extractor
        extractor_keywords = ['extractor', 'feature_extractor', 'featurizer']
        found_extractor_keywords = [k for k in extractor_keywords if k in args]

        if found_extractor_keywords:
            assert len(found_extractor_keywords) == 1, f"Multiple feature extractor arguments were given: {found_extractor_keywords}"

            self._set_feature_extractor(args[found_extractor_keywords[0]])
        