import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import time
from tqdm import tqdm
import model
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
from collections import OrderedDict
import pathlib
import os
import matplotlib.pyplot as pl
import threading
try:
    import telegram
    from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
    TELEGRAM_BOT = True
except:
    TELEGRAM_BOT = False

class TelegramBot(object):

    def __init__(self) -> None:
        self.token = '1288438831:AAFOJgulfJojpNGk4zRVrCj-bImqn6RcjAE'
        self.chat_id = '-468700262'

        self.updater = Updater(token=self.token)
        self.updater.dispatcher.add_handler(CommandHandler('stop', self.stop_training))

        self.updater.start_polling()
        self.bot_active = True
        
        print('Telegram bot started')

    def send_message(self, message):
        self.updater.bot.send_message(chat_id=self.chat_id, text=message)

    def send_image(self, image):
        pass
        # self.updater.bot.send_photo(chat_id=self.chat_id, photo=open(image, 'rb'))

    def stop_training(self, update, context):
        update.message.reply_text(" Stopping...")
        self.stop()
        sys.exit()

    def stop(self):
        os.kill(os.getpid(), signal.SIGINT)

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, n_training):
        """
        Very simple training set made of 200 Gaussians of width between 0.5 and 1.5
        We later augment this with a velocity and amplitude.
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Dataset, self).__init__()

        self.n_training = n_training
        
        x = np.linspace(-5.0, 5.0, 100)
        self.sigma = 1.0 * np.random.rand(200) + 0.5
        self.y = np.exp(-x[None,:]**2 / self.sigma[:,None]**2)

        self.indices = np.random.randint(low=0, high=200, size=self.n_training)
        self.amplitude = 2.0 * np.random.rand(self.n_training) + 0.1
        
    def __getitem__(self, index):

        amplitude = self.amplitude[index]
        sigma = self.sigma[self.indices[index]]

        inp = amplitude * self.y[self.indices[index],:]

        out = np.array([sigma, amplitude])

        return inp.astype('float32'), out.astype('float32')

    def __len__(self):
        return self.n_training
        

class Training(object):
    def __init__(self, hyperparameters):

        self.hyperparameters = hyperparameters

        self.cuda = torch.cuda.is_available()
        self.gpu = hyperparameters['gpu']
        self.smooth = hyperparameters['smooth']
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = hyperparameters['batch_size']        
                
        kwargs = {'num_workers': 2, 'pin_memory': False} if self.cuda else {}        
        
        self.model = model.Network(n_input=self.hyperparameters['n_input'],
                                n_hidden=self.hyperparameters['n_hidden'],
                                n_output=self.hyperparameters['n_output']).to(self.device)
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        option_training = 0

        #####################
        # Option 1. Training and validation sets separately
        #####################
        if (option_training == 0):
            self.train_dataset = Dataset(n_training=1000)
            self.validation_dataset = Dataset(n_training=100)
                    
            # Data loaders that will inject data during training
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, **kwargs)
            self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, **kwargs)


        #####################
        # Option 2. Only one training set from which we extract both training and validation
        #####################
        if (option_training == 1):
            self.dataset = Dataset(n_training=None)
            
            self.validation_split = hyperparameters['validation_split']
            idx = np.arange(self.dataset.n_training)
            
            self.train_index = idx[0:int((1-self.validation_split)*self.dataset.n_training)]
            self.validation_index = idx[int((1-self.validation_split)*self.dataset.n_training):]

            # Define samplers for the training and validation sets
            self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
            self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
                    
            # Data loaders that will inject data during training
            self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)
            self.validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)

        if (TELEGRAM_BOT):
            self.bot = TelegramBot()

    def init_optimize(self):

        self.lr = hyperparameters['lr']
        self.wd = hyperparameters['wd']
        self.n_epochs = hyperparameters['n_epochs']
        
        print('Learning rate : {0}'.format(self.lr))        
        
        p = pathlib.Path('weights/')
        p.mkdir(parents=True, exist_ok=True)

        current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
        self.out_name = 'weights/{0}'.format(current_time)

        # Copy model
        f = open(model.__file__, 'r')
        self.hyperparameters['model_code'] = f.readlines()
        f.close()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.loss_fn = nn.MSELoss().to(self.device)
        
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler, gamma=0.5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs, eta_min=0.1*self.lr)

    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = 1e100        
        
        print('Model : {0}'.format(self.out_name))

        for epoch in range(1, self.n_epochs + 1):            
            loss = self.train(epoch)
            loss_val = self.test()

            self.scheduler.step()

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
                'hyperparameters': self.hyperparameters
            }

            if (loss_val < best_loss):
                print(f"Saving model {self.out_name}.best.pth")                
                best_loss = loss_val
                torch.save(checkpoint, f'{self.out_name}.best.pth')

            if (self.hyperparameters['save_all_epochs']):
                torch.save(checkpoint, f'{self.out_name}.ep_{epoch}.pth')

        if (TELEGRAM_BOT):
            self.bot.stop()

    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (inputs, outputs) in enumerate(t):
            inputs = inputs.to(self.device)
            outputs = outputs.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(inputs)
            
            # Loss
            loss = self.loss_fn(out, outputs)
                    
            loss.backward()

            self.optimizer.step()

            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            if (NVIDIA_SMI):
                tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_usage = f'{tmp.gpu}'
                tmp = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                memory_usage = f' {tmp.used / tmp.total * 100.0:4.1f}'                
            else:
                gpu_usage = 'NA'
                memory_usage = 'NA'

            tmp = OrderedDict()
            tmp['gpu'] = gpu_usage
            tmp['mem'] = memory_usage
            tmp['lr'] = current_lr
            tmp['loss'] = loss_avg
            t.set_postfix(ordered_dict = tmp)
            
        self.loss.append(loss_avg)
        
        if (TELEGRAM_BOT):
            self.bot.send_message(f'Ep: {epoch} - L={loss_avg:7.4f}')
            pl.imsave('test.png', np.random.randn(100,100))
            self.bot.send_image('test.png')

        return loss_avg

    def test(self):
        self.model.eval()
        t = tqdm(self.validation_loader)
        loss_avg = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, outputs) in enumerate(t):
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                        
                out = self.model(inputs)
            
                # Loss
                loss = self.loss_fn(out, outputs)

                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                
                t.set_postfix(loss=loss_avg)
            
        return loss_avg

if (__name__ == '__main__'):

    hyperparameters = {
        'batch_size': 4096,
        'validation_split': 0.2,
        'gpu': 0,
        'lr': 3e-4,
        'wd': 0.0,
        'n_epochs': 10,
        'smooth': 0.15,
        'save_all_epochs': True,
        'n_input': 100,
        'n_hidden': 40,
        'n_output': 2
    }
    
    deepnet = Training(hyperparameters)
    deepnet.init_optimize()
    deepnet.optimize()