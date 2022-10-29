from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F




def get_logits(model, dataset, is_data_loader: bool = False, num_classes: int = 1000, batch_size: int = 128, num_workers: int = 4, gpu: Union[int, None] = None):
    r"""
    Args:
        model: Pretrained PyTorch model
        dataset: PyTorch dataset that has __len__ attribute
        num_classes: Number of classes
        batch_size: Batch size for the data loader
        num_workers: Number of workers for the data loader
        gpu: GPU index to use

    Returns:
        TensorDataset: Contains the model logits as samples and the original labels as the targets
    """

    if is_data_loader:
        assert isinstance(dataset, torch.utils.data.DataLoader), f"is_data_loader is set to True but dataset of type {type(dataset)} is not a supported data_loader"

        data_loader = dataset

        logits = torch.zeros((len(dataset.dataset), num_classes), dtype=torch.float)
        labels = torch.zeros((len(dataset.dataset),), dtype=torch.int)

    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                    batch_size=batch_size, shuffle=False,num_workers=num_workers, pin_memory=True
                                    )
        
        logits = torch.zeros((len(dataset), num_classes), dtype=torch.float)
        labels = torch.zeros((len(dataset),), dtype=torch.int)

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(data_loader)):

            if gpu is not None:
                input = input.cuda(gpu)
            target = target.cuda(gpu)

            output = model(input)

            logits[i * batch_size : i * batch_size + input.shape[0],:] = output.detach().cpu()
            labels[i * batch_size : i * batch_size + input.shape[0]] = target.detach().cpu()

    return torch.utils.data.TensorDataset(logits, labels)



def get_sorted_softmax(logits: torch.Tensor, temperature: Union[float, torch.Tensor, None] = None):
    r"""Computes softmax scores sorted along the class dimension"""

    if temperature is None:
        softmax_temp = torch.Tensor([1.0])
    elif isinstance(temperature, float):
        softmax_temp = torch.Tensor([temperature])
    elif isinstance(temperature, torch.Tensor):
        softmax_temp = temperature
    else:
        raise TypeError(f"{type(temperature)} is not supported for setting a softmax temperature!")
    
    softmax_scores = F.softmax(logits / softmax_temp, dim=1)

    softmax_scores, preds = softmax_scores.sort(dim=1, descending=True)
    softmax_scores = softmax_scores.cumsum(dim=1)

    return softmax_scores, preds




def get_class_confidence_accuracy(logit_dataset, num_classes=None, by_predictions=False, replace_nans=False):
    r"""
    Return class-wise analytics.
    args:
        logit_dataset: Torch TensorDataset containing (logits, label) tuples per sample.
    returns:
        avg_confidence (torch.Tensor): 1-dimensional tensor containing average top-1 confidence.
        accuracy (torch.Tensor): 1-dimensional tensor containing accuracy.
    """
    if num_classes:
        class_list = list(range(num_classes))
    else:
        # get class list from the targets since predictions might not contain every class
        class_list = logit_dataset.tensors[1].unique()

    softmax_scores, preds = get_sorted_softmax(logit_dataset.tensors[0])

    confs = torch.zeros(len(class_list))
    accs = torch.zeros(len(class_list))
    for cur_target in class_list:

        if by_predictions:
            cur_idx = preds[:,0] == cur_target
        else:
            cur_idx = logit_dataset.tensors[1] == cur_target

        if replace_nans and (cur_idx.sum() < 1):
            confs[cur_target] = 0.0
            accs[cur_target] = 0.0
        else:
            confs[cur_target] = softmax_scores[cur_idx, 0].sum() / cur_idx.sum()
            accs[cur_target] = (logit_dataset.tensors[1][cur_idx] == preds[cur_idx, 0]).sum() / cur_idx.sum()

    return confs, accs



def confidence_histogram_with_accuracy(logit_dataset, bins=None):
    r"""
    Return a histogram of the model confidence across the dataset with classification accuracy information for each bin.
    args:
        logit_dataset: Torch TensorDataset containing (logits, label) tuples per sample.
    returns:
        num_samples (torch.Tensor): 1-dimensional tensor containing the number of samples that fall onto each bin.
        bins (torch.Tensor): 1-dimensional tensor containing the bin edges for model confidence
        accuracy (torch.Tensor): 1-dimensional tensor containing classification accuracy at each histogram bin.
    """
    softmax_range = [0.0, 1.01]

    softmax_scores, preds = get_sorted_softmax(logit_dataset.tensors[0])

    num_samples, hist_bins = torch.histogram(softmax_scores[:,0], bins=bins, range=softmax_range)

    # compute classification accuracy at each confidence level
    correct = preds[:,0].eq(logit_dataset.tensors[1])

    acc_hist = binned_statistic(softmax_scores[:,0].detach().cpu().numpy(), 
                                correct.detach().cpu().numpy(), 
                                lambda x: np.mean(x), 
                                bins, 
                                softmax_range)

    return num_samples, acc_hist, hist_bins
    


def binned_statistic(x, values, func, nbins, range):
    r"""The usage is nearly the same as scipy.stats.binned_statistic"""
    from scipy.sparse import csr_matrix

    N = len(values)
    r0, r1 = range

    digitized = (float(nbins)/(r1 - r0)*(x - r0)).astype(int)
    S = csr_matrix((values, [digitized, np.arange(N)]), shape=(nbins, N))

    return [func(group) for group in np.split(S.data, S.indptr[1:-1])]



def platt_scaling(logits_dataset: Any, init_value: float = 1.3, max_iters: int = 10, lr: float = 0.01, epsilon: float = 0.01, is_data_loader: bool = False, batch_size: int = 128, num_workers: int = 4, gpu: Union[int, None] = None):
    r"""
    Args:
        logits_dataset: PyTorch dataset that returns ``logits, targets`` tuples.
        lr (float): learning rate (default: `0.01`).
        epsilon (float): Tolerance in ``temperature`` updates used to terminate optimization (default: 0.01).
        batch_size: Batch size for the data loader.
        num_workers: Number of workers for the data loader.
        gpu: GPU index to use

    Returns:
        Temperature (float): Scalar temperature value found by calibration.
    """

    if is_data_loader:
        assert isinstance(logits_dataset, torch.utils.data.DataLoader), f"is_data_loader is set to True but dataset of type {type(logits_dataset)} is not a supported data_loader"

        data_loader = logits_dataset

    else:
        data_loader = torch.utils.data.DataLoader(logits_dataset,
                                    batch_size=batch_size, shuffle=False,num_workers=num_workers, pin_memory=True
                                    )
    
    
    criterion = nn.CrossEntropyLoss()
    tmp = torch.Tensor([init_value])

    if gpu is not None:
        tmp = tmp.cuda(gpu)
        criterion = criterion.cuda(gpu)

    tmp.requires_grad_(True)
    optimizer = torch.optim.SGD([tmp], lr)

    for iter in range(max_iters):
        tmp_old = tmp.item()

        for logit, target in data_loader:

            target = target.to(torch.long)
            if gpu is not None:
                logit = logit.cuda(gpu)
                target = target.cuda(gpu)

            output = logit / tmp

            loss = criterion(output, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        if abs(tmp_old - tmp.item()) < epsilon:
            break

    return tmp.detach().cpu()

