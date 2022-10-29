from typing import Any, Union
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from ..pytorch_utils.inference import get_logits, platt_scaling
from ..pytorch_utils.train_utils import AverageMeter


class ConformalModel:

    def __init__(self, 
                model, 
                dataset,
                is_logits: bool = False,
                is_softmax: bool = False,
                num_classes: int = 1000,
                alpha: float = 0.1, 
                randomized: bool = True,
                allow_zero_sets: bool = False,
                use_platt_scaling: Union[float, torch.Tensor, bool] = False,
                kreg: int = 0,
                lamda: float = 0.0,
                batch_size: int = 128,
                num_workers: int = 4,
                gpu: Union[None, int] = None) -> None:
        
        self.model = model.cuda(gpu)
        self.alpha = alpha

        # data args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gpu = gpu

        # conformal calibration args
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets

        # RAPS parameters
        self.kreg = kreg
        self.lamda = lamda

        assert is_logits >= is_softmax, "If scores are given, please also set is_logits to True"
        self.is_softmax = is_softmax
        
        if is_logits:
            calib_logits = dataset
        else:
            # get logits for the calibration samples
            print('Computing logits for the calibration set...')
            calib_logits = get_logits(
                model,
                dataset,
                num_classes=num_classes,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                gpu=self.gpu
            )

        # platt scaling
        if isinstance(use_platt_scaling, bool):
            if use_platt_scaling is True:
                self.softmax_temp = platt_scaling(calib_logits, batch_size=self.batch_size, num_workers=self.num_workers, gpu=self.gpu)
            else:
                self.softmax_temp = torch.Tensor([1.0])  # no scaling
        elif isinstance(use_platt_scaling, float):
            self.softmax_temp = torch.Tensor([use_platt_scaling])
        elif isinstance(use_platt_scaling, torch.Tensor):
            self.softmax_temp = use_platt_scaling
        else:
            raise TypeError(f"{type(use_platt_scaling)} is not supported!")
        
        # compute Q_hat
        self.q_hat = self._get_Q_hat(calib_logits)


    def validate_conformal(self, val_set, is_logits=False, is_softmax=False, **kwargs):
        r"""
        args:
            val_set: validation dataset
        returns:
            top1 (AverageMeter): Accuracy@1
            top5 (AverageMeter): Accuracy@5
            coverage (AverageMeter): Average coverage
            size (AverageMeter): Average size of the confidence sets
        """

        # use custom accuracy and coverage computing functions if given, else use the standard class static functions
        if 'acc_fn' in kwargs and kwargs['acc_fn'] is not None:
            acc_fn = kwargs['acc_fn']
        else:
            acc_fn = self._accuracy
        
        if 'cov_fn' in kwargs and kwargs['cov_fn'] is not None:
            cov_fn = kwargs['cov_fn']
        else:
            cov_fn = self._get_coverage


        # conformal prediction parameters
        randomized = kwargs['randomized'] if 'randomized' in kwargs else self.randomized
        allow_zero_sets = kwargs['allow_zero_sets'] if 'allow_zero_sets' in kwargs else self.allow_zero_sets

        get_set_stats = kwargs["get_set_stats"] if "get_set_stats" in kwargs else False 


        assert is_logits >= is_softmax, "If scores are given, please also set is_logits to True"

        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage = AverageMeter('RAPS coverage')
        size = AverageMeter('RAPS size')
        if get_set_stats:
            stats = Counter()

        # if logits are not given, gel logits using the attribute model
        if is_logits:
            logit_set = val_set
        else:
            logit_set = get_logits(self.model, val_set, 
                            batch_size=self.batch_size, 
                            num_workers=self.num_workers, 
                            gpu=self.gpu)

        val_loader = torch.utils.data.DataLoader(logit_set,
                                batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers, pin_memory=True
                                )


        with tqdm(total=len(val_loader), position=1, bar_format='{desc}', desc='Val metrics') as bmetric:
            for i, (input, target) in enumerate(tqdm(val_loader, position=0)):
                # do a validation step which returns a dict of metric values for the batch
                batch_val = self.validation_step(batch=(input, target), batch_idx=i, acc_fn=acc_fn, cov_fn=cov_fn, randomized=randomized, allow_zero_sets=allow_zero_sets, is_softmax=is_softmax, get_set_stats=get_set_stats)

                # update meters
                top1.update(batch_val['acc1'], input.shape[0])
                top5.update(batch_val['acc5'], input.shape[0])
                coverage.update(batch_val['coverage'], input.shape[0])
                size.update(batch_val['size'], input.shape[0])
                if get_set_stats:
                    stats.update(batch_val["set_stats"])

                bmetric.set_description_str(
                        'Acc@1: {acc1_batch:.3f} ({acc1:.3f}) | '.format(acc1_batch=batch_val['acc1'], acc1=top1.avg)
                        + 'Acc@5: {acc5_batch:.3f} ({acc5:.3f}) | '.format(acc5_batch=batch_val['acc5'], acc5=top5.avg)
                        + 'Coverage: {cov_batch:.3f} ({cov:.3f}) | '.format(cov_batch=batch_val['coverage'], cov=coverage.avg)
                        + 'Size: {size_batch:.3f} ({size:.3f}) | '.format(size_batch=batch_val['size'], size=size.avg)
                    )

        if get_set_stats:
            return top1.avg, top5.avg, coverage.avg, size.avg, stats

        return top1.avg, top5.avg, coverage.avg, size.avg



    def validation_step(self, batch, batch_idx, acc_fn, cov_fn, randomized, allow_zero_sets, is_softmax=False, get_set_stats=False):
        """Make a validation step"""

        output, target = batch
        
        acc1, acc5 = acc_fn(output, target, topk=(1, min(5, output.size(1)) ))

        set_C = self.prediction_set_batch(output, randomized, allow_zero_sets, is_softmax)

        coverage, size = cov_fn(set_C, target.detach().cpu().numpy())

        batch_dict = {
            'acc1': acc1.item()/100,
            'acc5': acc5.item()/100,
            'coverage': coverage,
            'size': size
        }

        if get_set_stats:
            batch_dict.update({"set_stats": Counter([ len(sc) for sc in set_C ])})

        return batch_dict



    def prediction_set_batch(self, batch_logits, randomized, allow_zero_sets, is_softmax=False):
        r"""
        args:
            batch_logits: Raw model output logits for the entire batch
        returns:
            c_set: Confidence set that yields valid marginal coverage :math:`1 - \alpha`

        Computes confidence sets for the whole batch based on :math:`\hat{Q}` found via calibration.
        """

        if not is_softmax:
            batch_softmax_scores = F.softmax(batch_logits / self.softmax_temp, dim=1)
        else:
            batch_softmax_scores = batch_logits

        batch_softmax_scores = batch_softmax_scores.detach().cpu().numpy()

        set_C = [self._compute_S(s, self.q_hat, randomized, allow_zero_sets) for s in batch_softmax_scores]

        return set_C



    def _get_Q_hat(self, calib_logits):
        r"""
        Computes :math:`\hat{Q}_{1-\alpha} (\{ E_i \}_{i \in {1,2,\ldots,n}})` as in step 6 of Algorithm 1 of `Romano et al. (2020)`__

        :math:`\hat{Q}_{1-\alpha}` is the :math:`(1-\alpha)(1+n)`th quantile of E.

        __ https://arxiv.org/pdf/2006.02544.pdf
        """

        calib_loader = torch.utils.data.DataLoader(calib_logits,
                                batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers, pin_memory=True
                                )

        e_score_all = np.zeros(len(calib_loader.dataset))
        for i, (logits, target) in enumerate(calib_loader):

            if not self.is_softmax:
                batch_softmax_scores = F.softmax(logits / self.softmax_temp, dim=1)
            else:
                batch_softmax_scores = logits
            batch_softmax_scores = batch_softmax_scores.numpy()
            
            target = target.numpy()

            e_score_batch = [self.get_tau(ss, t) for ss, t in zip(batch_softmax_scores, target)]

            e_score_all[i * self.batch_size : i * self.batch_size + logits.shape[0]] = np.array(e_score_batch)

        q_hat = np.quantile(e_score_all, (1 - self.alpha)*(1 + 1/len(calib_loader.dataset)))

        return q_hat



    def get_tau(self, sample_softmax_scores, sample_target):
        r"""
        args:
            sample_softmax_scores: Unordered model probability estimates where indices directly imply the class
            sample_target: Correct class of the sample
        returns:
            tau 

        :math:`\tau` in eq. (7) of `Romano et al. (2020)`__

        If not randomized, we would have :math:`\tau = \pi_1 + \pi_2 + ... + \pi_{c^\ast}`.

        On the other hand, if we randomize, then we would be removing the correct class depending on the realization of random :math:`u`. Therefore, with the same probability we will choose :math:`\tau = \pi_1 + \pi_2 + ... + \pi_{c^\ast} + \pi_{c^\ast + 1}`.

        Note that `+ max(0, top_c - (self.kreg-1)) * self.lamda` adds regularization directly to :math:`\tau`.

        __ https://arxiv.org/pdf/2006.02544.pdf
        """
        
        # remove redundant axes
        sample_softmax_scores = np.squeeze(sample_softmax_scores)

        # sort logits if not sorted, sort_idx elements correspond to classes
        sort_idx = np.argsort( -sample_softmax_scores )
        sample_softmax_scores = sample_softmax_scores[sort_idx]

        top_c = np.argwhere( sort_idx == sample_target ).item()

        # not including top_c:
        tau = sample_softmax_scores[:top_c].sum()
        
        # randomized with u
        if self.randomized:
            u_rand = np.random.random()

            # if top_c is 0, i.e., top predicted class is the correct target and we do not allow sets of size zero, we can skip randomization for this sample
            if (not self.allow_zero_sets) and (top_c == 0):
                tau += sample_softmax_scores[top_c] + max(0, top_c - (self.kreg-1)) * self.lamda
            else:
                tau += u_rand * sample_softmax_scores[top_c] + max(0, top_c - (self.kreg-1)) * self.lamda
        else:
            tau += sample_softmax_scores[top_c] + max(0, top_c - (self.kreg-1)) * self.lamda

        return tau



    def _compute_S(self, sample_softmax_scores, tau, randomized, allow_zero_sets):
        r"""
        args:
            sample_softmax_scores: Raw model probability estimates where indices directly imply the class
            tau: target probability
        returns:
            randomized (depending on class attribute) confidence sets

        :math:`S(x, u; \pi, \tau)` in eq. (5) of `Romano et al. (2020)`__


        __ https://arxiv.org/pdf/2006.02544.pdf
        """

        # remove redundant axes
        sample_softmax_scores = np.squeeze(sample_softmax_scores)

        # sort logits if not sorted, sort_idx elements correspond to classes
        sort_idx = np.argsort( -sample_softmax_scores )
        sample_softmax_scores = sample_softmax_scores[sort_idx]


        # compute L(x,pi,tau)
        l_min = self._compute_L(sample_softmax_scores, tau)

        # compute V(x,pi,tau)
        v_score = self._compute_V(sample_softmax_scores, tau, l_min)

        set_S = sort_idx[:l_min]

        # randomized with u
        if randomized:

            u_rand = np.random.random()
            if u_rand <= v_score:
                # if l_min is just 1 and we do not allow sets of size zero, we can skip randomization for this sample
                if (not allow_zero_sets) and (l_min == 1):
                    pass
                else:
                    set_S = set_S[:-1]
        
        return set_S


    
    def _compute_V(self, sample_softmax_scores, tau, l_min):
        r"""
        args:
            sample_softmax_scores: Sorted (decreasing) model probability estimates
            tau: target probability
            l_min: minimum index found by :math:`L(x; \pi, \tau)` in eq. (3) of `Romano et al. (2020)`__
        returns:
            :math:`V(x; \pi, \tau)` n eq. (5) of `Romano et al. (2020)`__


        __ https://arxiv.org/pdf/2006.02544.pdf
        """

        # record the l_min'th class prob. separately to prevent added lambda penalty
        v_denom = sample_softmax_scores[min(l_min, len(sample_softmax_scores)) - 1]   # to prevent index error if l_min is too large

        # add regularization to the prob. estimates:
        sample_softmax_scores[self.kreg:] += self.lamda

        return (sample_softmax_scores[:l_min].sum() - tau) / v_denom



    def _compute_L(self, sample_softmax_scores, tau):
        r"""
        args:
            sample_softmax_scores: Sorted (decreasing) model probability estimates
            tau: target probability
        returns:
            minimum index for which the cumulative sum of probability estimates from the top exceeds tau

        Computes cumulative sum and finds the index :math:`i`, where

        .. math::
            \begin{aligned}
                \pi_1 + \pi_2 + ... + \pi_{i-1} < \tau
                \pi_1 + \pi_2 + ... + \pi_{i-1} + \pi_i \geq \tau
            \end{aligned}
        """

        # add regularization to the prob. estimates:
        sample_softmax_scores[self.kreg:] += self.lamda

        return sum(np.cumsum(sample_softmax_scores) < tau) + 1



    def _get_coverage(self, set_C, targets):
        
        covered = np.zeros(len(set_C))
        sizes = np.zeros(len(set_C))
        for i, (cur_set, cur_target) in enumerate(zip(set_C, targets)):

            if cur_target in cur_set:
                covered[i] = 1
            sizes[i] = cur_set.shape[0]

        return covered.mean(), sizes.mean()



    def _accuracy(self, output, target, topk=(1,), track=False):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    
    def __get_coverage(self, set_C, targets):
        print(DeprecationWarning("Double underscore method names are deprecated for this class. Please use the single underscore version!"))
        return self._get_coverage(set_C, targets)
    
    def __accuracy(self, output, target, topk=(1,), track=False):
        print(DeprecationWarning("Double underscore method names are deprecated for this class. Please use the single underscore version!"))
        return self._get_coverage(output, target, topk=(1,), track=False)



class ThresholdedConformalModel(ConformalModel):

    def __init__(self, model: nn.Sequential, dataset: Any, is_logits: bool = False, is_softmax: bool = False, num_classes: int = 1000, alpha: float = 0.1, randomized: bool = True, allow_zero_sets: bool = True, use_platt_scaling: bool = False, kreg: int = 0, lamda: float = 0.0, batch_size: int = 128, num_workers: int = 4, gpu: int = None) -> None:
        r"""
        Thresholded Conformal Prediction (TCP, THR, LAC...) from `Sadinle et al. (2019)`__

        Args:
            model (nn.Sequential): Classifier. Pytorch model.


        __ https://arxiv.org/pdf/1609.00451.pdf
        """

        super().__init__(model, dataset, is_logits, is_softmax, num_classes, alpha, randomized, allow_zero_sets, use_platt_scaling, kreg, lamda, batch_size, num_workers, gpu)




    def get_tau(self, sample_softmax_scores: np.ndarray, sample_target: int) -> float:
        r"""
        Args:
            sample_softmax_scores: Unordered model probability estimates where indices directly imply the class
            sample_target: Correct class of the sample

        Returns:
            1 - tau

        :math:`\tau = \pi_{\text{sample_target}} (x)` is used directly as tau for TCP.
        """
        
        # remove redundant axes
        sample_softmax_scores = np.squeeze(sample_softmax_scores)

        return 1.0 - sample_softmax_scores[sample_target]



    def _compute_S(self, sample_softmax_scores: np.ndarray, tau, randomized, allow_zero_sets) -> np.ndarray:
        r"""
        Args:
            sample_softmax_scores: Raw model probability estimates where indices directly imply the class
            tau: target probability

        Returns:
            confidence set for the sample

        :math:`S(x, u; \pi, \tau)` in eq. (5) of `Romano et al. (2020)`__


        __ https://arxiv.org/pdf/2006.02544.pdf
        """

        # remove redundant axes
        sample_softmax_scores = np.squeeze(sample_softmax_scores)

        set_C_idx_tuple = np.nonzero(sample_softmax_scores >= 1.0 - tau)

        return set_C_idx_tuple[0]
        