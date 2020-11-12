import random
import time, datetime
import sys, os

import torch
from torch.autograd import Variable


class Logger():
    def __init__(self, epoch, n_epochs, batches_epoch):  # batches_epoch is the variable for number of batches per epoch, which
                                            # is same as the number of images the dataloader extracts for each batch
        # self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = epoch
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}

    def log(self, save_path, losses=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                # self.losses[loss_name] = losses[loss_name].data[0]
                self.losses[loss_name] = losses[loss_name].item()
            else:
                # self.losses[loss_name] += losses[loss_name].data[0]
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch-self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # At the end of an epoch
        if (self.batch % self.batches_epoch) == 0:
            log_epoch = 'Epoch %03d/%03d -- ' % (self.epoch, self.n_epochs)

            batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
            batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
            log_eta = 'ETA: %s -- ' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done))

            with open(os.path.join(save_path, 'log_stats.txt'), 'a') as f_:
                f_.write(log_epoch + log_eta)

            for loss_name, loss in self.losses.items():
                log_losses = '%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch)
                with open(os.path.join(save_path, 'log_stats.txt'), 'a') as f_:
                    f_.write(log_losses)

                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            with open(os.path.join(save_path, 'log_stats.txt'), 'a') as f_:
                f_.write('\n')

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Learning rate decay must start before the training ends!!!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    # The paper mentions: "Weights are initialized from a Gaussian distribution N (0, 0.02)".
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)