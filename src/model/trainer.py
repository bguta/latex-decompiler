import os

import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm, trange

class Trainer():
    """
    A trainer for the neural network on the gpu

    # Arguments

    # Example
    """
    def __init__(self,
                optimizer,
                loss_fn,
                model,
                train_generator,
                val_generator,
                model_path,
                lr_scheduler=None,
                init_epoch=1,
                epsilon=0.95,
                best_val_loss=np.inf,
                num_epochs=10):
            
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.model_path = model_path

        self.epoch = init_epoch
        self.final_epoch = self.epoch + num_epochs
        self.step = 0
        self.total_step = 0
        self.best_val_loss = best_val_loss
        self.device = torch.device('cuda')
        self.epsilon = epsilon
        self.model.to(self.device)
    
    def train(self):
        epoch_stats = "Epoch {}, step: {}/{} {:.2f}%, Loss: {:.4f}"
        print('Starting to Train')
        print_freq = 32

        while self.epoch <= self.final_epoch:
            self.model.train()
            losses = 0.0
            #with click.progressbar(range(len(self.train_generator)), label=f'Epoch: {self.epoch}/{self.final_epoch}') as bar:
            loop = trange(len(self.train_generator), ascii=" #")
            loop.set_description(f'Epoch: {self.epoch}/{self.final_epoch}')
            for index in loop:
                imgs, targets, loss_targets = self.train_generator.__getitem__(index)
                step_loss = self.train_step(imgs, targets, loss_targets)
                losses += step_loss
                avg_loss = losses/(index+1)
                loop.set_postfix(loss=avg_loss)
                # log message
                # if self.step % print_freq == 0:
                #     #self.epsilon *= 0.95
                #     avg_loss = losses / print_freq
                #     # print(epoch_stats.format(
                #     #     self.epoch, self.step, len(self.train_generator),
                #     #     100 * self.step / len(self.train_generator),
                #     #     avg_loss
                #     # ))
                #     losses = 0.0
            self.train_generator.on_epoch_end()
            # calc val
            val_loss = self.validate()
            # self.lr_schedule.step(val_loss)

            self.save_model('ckpt-{}-{:.4f}.pt'.format(self.epoch, val_loss))
            self.epoch += 1
            self.step = 0
        return
    def validate(self):
        self.model.eval()
        val_total_loss = 0.0
        epoch_stats = "Epoch {}, validation loss: {:.4f}"

        # run the prediction with no grad acumulation
        with torch.no_grad():
            for index in range(len(self.val_generator)):
                imgs, target, loss_target = self.val_generator.__getitem__(index)
                val_total_loss += self.val_step(imgs, target, loss_target)
            self.val_generator.on_epoch_end()
            avg_loss = val_total_loss / len(self.val_generator)
            print(epoch_stats.format(self.epoch, avg_loss))

        if avg_loss < self.best_val_loss:
            print('val loss improved from {:.4f} to {:.4f}'.format(self.best_val_loss, avg_loss))
            self.epsilon = self.epsilon*0.95
            self.best_val_loss = avg_loss
            self.save_model('best_ckpt.pt')
        return avg_loss
    def train_step(self, imgs, targets, loss_targets):
        self.optimizer.zero_grad()
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        loss_targets = loss_targets.to(self.device)
        logits = self.model(imgs, targets, self.epsilon)

        # calculate the loss
        loss = self.loss_fn(loss_targets, logits)
        loss.backward()
        self.step += 1
        self.total_step += 1
        self.optimizer.step()

        return loss.item()

    def val_step(self, imgs, targets, loss_target):
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        loss_target = loss_target.to(self.device)
        logits = self.model(imgs, targets, self.epsilon)

        # calculate loss
        loss = self.loss_fn(loss_target, logits)

        return loss
    def save_model(self, model_name):
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        save_path = os.path.join(self.model_path, model_name)

        print("Saving checkpoint to {}".format(save_path))

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epsilon': self.epsilon
        }, save_path)

