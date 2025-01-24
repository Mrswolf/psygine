# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2022/06/15
# License: MIT License
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from skorch.classifier import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, EpochScoring, Checkpoint, EarlyStopping


class NeuralNetClassifierNoLog(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        return super(NeuralNetClassifier, self).get_loss(y_pred, y_true, *args, **kwargs)

    def fit(self, X, y, **fit_params):
        net = super(NeuralNetClassifier, self).fit(X, y, **fit_params)
        callbacks = OrderedDict(net.callbacks)
        if 'checkpoint' in callbacks:
            net.load_params(
                checkpoint=callbacks['checkpoint'])
        return net


class SkorchNet:
    def __init__(self, module):
        self.module = module
    
    def __call__(self, *args, **kwargs):
        model = self.module(*args, **kwargs)
        net = NeuralNetClassifierNoLog(model,
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            optimizer__weight_decay=0,
            batch_size=128, 
            lr=1e-2, 
            max_epochs=300,
            device="cpu",
            train_split=ValidSplit(0.2, stratified=True),
            iterator_train__shuffle=True,
            callbacks=[
                ('train_acc', EpochScoring('accuracy', 
                                            name='train_acc', 
                                            on_train=True, 
                                            lower_is_better=False)),
                ('lr_scheduler', LRScheduler('CosineAnnealingLR', T_max=300 - 1)),
                ('estoper', EarlyStopping(patience=50)),
                ('checkpoint', Checkpoint(dirname="checkpoints/{:s}".format(str(id(model))))),
            ],
            verbose=True)
        return net