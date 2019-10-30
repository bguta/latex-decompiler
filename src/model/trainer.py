import os
from os.path import join

import torch
from torch.nn.utils import clip_grad_norm_

class Trainer(object):
    '''
    A trainer for the neural network on the gpu
    '''
    def __init__(self,
                optimizer,
                model,
                lr_schedule,
                train_generator,
                val_generator,
                init_epoch=1,
                num_epochs=10)
            
        self.optimizer = optimizer
        self.model = model
        self.lr_schedule = lr_schedule
        self.train_generator = train_generator

        self.epoch = init_epoch
        self.final_epoch = self.epoch + num_epochs
        self.step = 0
        self.current_best_loss = 1e18
        self.device = torch.device('cude')
    
    def train(self):
        return
    def train_step(self, imgs, targets):
        return
    def val_step(self, imgs, targets):
        return
    def save_model(self, model_name):
        return
