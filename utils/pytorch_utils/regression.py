from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from .train_utils import save_checkpoint





def mlp(feature_dim: int = 10, num_outputs: int = 1, num_layers: int = 4, hidden_dim: int = 512):

    layer1 = nn.Linear(feature_dim, hidden_dim)
    if num_layers == 1:
        return layer1

    layers = [layer1]
    cur_dim = hidden_dim
    # hidden layers
    for i in range(num_layers-2):
        # add activations to the previous layer
        layers.append(nn.ReLU())
        # add new hidden layer
        layers.append(nn.Linear(cur_dim, int(cur_dim/2)))
        cur_dim = int(cur_dim/2)
    
    layers.append(nn.ReLU())
    layers.append(nn.Linear(cur_dim, num_outputs))

    return nn.Sequential(*layers)



def fit_regressor(Xs, ys, lr, weight_decay, momentum, max_iter, gpu, outdir, freq=20):
    print(DeprecationWarning("This function is deprecated."))

    # get MLP network
    model = mlp(feature_dim=Xs.shape[1])
    
    if gpu is not None:
        model = model.cuda(gpu)

    
    # loss
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    
    losses = []
    pbar = tqdm(range(max_iter), miniters=freq)

    model.train()
    for t in pbar:

        # forward pass
        y_pred = model(Xs)

        # compute loss
        loss = loss_fn(y_pred, ys)
        losses.append(loss.item())

        # update progress bar
        pbar.set_postfix({"loss": loss.item()})

        # gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save model checkpoint periodically
        if t % freq == 0:
            save_checkpoint({
                'iterations': t + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, False, outdir=outdir)

            # save losses
            np.save(outdir / 'loss.npy', np.array(losses))


    return model, losses

