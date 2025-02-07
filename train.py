import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import time
from tqdm import tqdm
import model
try:
    from nvitop import Device
    NVITOP = True
except:
    NVITOP = False
from collections import OrderedDict
import pathlib
import matplotlib.pyplot as pl
try:    
    from telegram.ext import ApplicationBuilder, CommandHandler
    import asyncio
    TELEGRAM_BOT = True
except:
    TELEGRAM_BOT = False

from matplotlib.lines import Line2D
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
        # if(p.requires_grad):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    pl.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    pl.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    pl.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    pl.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    pl.xlim(left=0, right=len(ave_grads))
    pl.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    pl.xlabel("Layers")
    pl.ylabel("average gradient")
    pl.title("Gradient flow")
    pl.grid(True)
    pl.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

class TelegramBot(object):
    def __init__(self):
        self.token = os.environ['TELEGRAM_TOKEN']
        self.chat_id = os.environ['TELEGRAM_CHATID']
                
    async def sendmessage(self, text):        
        await self.application.bot.sendMessage(chat_id=self.chat_id, text=text)

    def stop_training(self, update, context):
        self.bot_active = False
        update.message.reply_text("Stopping training...")
        self.stop()
        sys.exit()

    def send_message(self, text):
        self.application = ApplicationBuilder().token(self.token).build()   
        self.application.add_handler(CommandHandler('stop', self.stop_training))
        asyncio.run(self.sendmessage(text))
        
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
        self.use_bot = hyperparameters['bot']

        if (NVITOP):            
            self.handle = Device.all()[self.gpu]
            print(f"Computing in {self.device} : {self.handle.name()}")
        
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

        if (TELEGRAM_BOT and self.use_bot):
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

        self.n_batches = len(self.train_loader)
        self.warmup = hyperparameters['warmup']

        print(f"N. batches : {self.n_batches}")
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_batches * (self.n_epochs - self.warmup), eta_min=0.1*self.lr)
        warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=self.warmup * self.n_batches)

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup, scheduler], milestones=[self.warmup * self.n_batches])

    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = 1e100
        
        print('Model : {0}'.format(self.out_name))

        for epoch in range(1, self.n_epochs + 1):            
            loss = self.train(epoch)
            loss_val = self.test()            

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
                'hyperparameters': self.hyperparameters
            }

            torch.save(checkpoint, f'{self.out_name}.pth')

            if (loss_val < best_loss):
                print(f"Saving model {self.out_name}.best.pth")                
                best_loss = loss_val
                torch.save(checkpoint, f'{self.out_name}.best.pth')

            if (self.hyperparameters['save_all_epochs']):
                torch.save(checkpoint, f'{self.out_name}.ep_{epoch}.pth')


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

            # plot_grad_flow(self.model.named_parameters())

            # breakpoint()

            self.optimizer.step()

            self.scheduler.step()

            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            if (NVITOP):                
                gpu_usage = f'{self.handle.gpu_utilization()}'
                memory_usage = f'{self.handle.memory_used_human()}/{self.handle.memory_total_human()}'
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
        
        if (TELEGRAM_BOT and self.use_bot):            
            self.bot.send_message(f'Model : {self.out_name}\nEp: {epoch} - L={loss_avg:7.4f}')
            # pl.imsave('test.png', np.random.randn(100,100))
            # self.bot.send_image('test.png')

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
        'batch_size': 64,
        'validation_split': 0.2,
        'gpu': 0,
        'lr': 3e-4,
        'wd': 0.0,
        'n_epochs': 10,
        'smooth': 0.15,
        'save_all_epochs': True,
        'n_input': 100,
        'n_hidden': 40,
        'n_output': 2,
        'warmup': 5,
        'bot': False
    }
    
    deepnet = Training(hyperparameters)
    deepnet.init_optimize()
    deepnet.optimize()