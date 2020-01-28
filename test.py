import shutil
import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.functional as tf
import torch.utils.data
import time
from tqdm import tqdm
import model

class Testing(object):
    def __init__(self, gpu=0, checkpoint=None):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (checkpoint is None):
            files = glob.glob('trained/*.pth')
            self.checkpoint = max(files, key=os.path.getctime)
        else:
            self.checkpoint = '{0}'.format(checkpoint)
        
        self.model = model.Network(100, 40, 2).to(self.device)
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        print("=> loading checkpoint '{}'".format(self.checkpoint))

        checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(self.checkpoint))

    def test(self):
        self.model.eval()

        x = np.linspace(-5.0, 5.0, 100)
        sigma = 1.0 * np.random.rand(200) + 0.5
        amplitude = 2.0 * np.random.rand(200) + 0.1
        y = amplitude[:,None] * np.exp(-x[None,:]**2 / sigma[:,None]**2)
        
        with torch.no_grad():

            inputs = torch.tensor(y.astype('float32'))
            
            out = self.model(inputs)

        f, ax = pl.subplots(nrows=2, ncols=1)
        ax[0].plot(sigma, out[:,0], '.')
        ax[0].plot([0.2,1.7], [0.2,1.7])
        ax[1].plot(amplitude, out[:,1], '.')
        ax[1].plot([0.05,2.2], [0.05,2.2])

        ax[0].set_xlabel('sigma_orig')
        ax[0].set_ylabel('sigma_NN')
        ax[1].set_xlabel('amp_orig')
        ax[1].set_ylabel('amp_NN')
        pl.show()
       
            
if (__name__ == '__main__'):
    
    deepnet = Testing(gpu=0, checkpoint='trained/2020-01-28-11:38:50.pth')

    deepnet.test()