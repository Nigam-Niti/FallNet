
# Import required modules ...

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import get_video_backend
from torchvision.models.video import r3d_18 
from torchvision import transforms
import os
from tqdm.auto import tqdm
import numpy as np
import time
import av
import random
import train
import test
import SupervisionNet
import i3d
print(f"PyAV version -- {av.__version__}")

SEED = 491
torch.manual_seed(SEED)

from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

bs = 4
lr = 1e-2
gamma = 0.7
total_epochs = 10
config = {}
num_workers = 0

kwargs = {'num_workers':num_workers, 'pin_memory':True} if torch.cuda.is_available() else {'num_workers':num_workers}
#kwargs = {'num_workers':num_workers}
#kwargs = {}

hmdb51_train_v1, hmdb51_val_v1 = random_split(hmdb51_train, [total_train_samples - total_val_samples,
                                                                       total_val_samples])

#hmdb51_train_v1.video_clips.compute_clips(16, 1, frame_rate=30)
#hmdb51_val_v1.video_clips.compute_clips(16, 1, frame_rate=30)
#hmdb51_test.video_clips.compute_clips(16, 1, frame_rate=30)

#train_sampler = RandomClipSampler(hmdb51_train_v1.video_clips, 5)
#test_sampler = UniformClipSampler(hmdb51_test.video_clips, 5)
  
train_loader = DataLoader(hmdb51_train_v1, batch_size=bs, shuffle=True, **kwargs)
val_loader   = DataLoader(hmdb51_val_v1, batch_size=bs, shuffle=True, **kwargs)
test_loader  = DataLoader(hmdb51_test, batch_size=bs, shuffle=False, **kwargs)

model = SupervisionNet()

if torch.cuda.is_available():
   model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

print("Launching Action Recognition Model training")
for epoch in range(1, total_epochs + 1):
    print(train_loader)
    train(config, model, train_loader, optimizer, epoch)
    test(config, model, val_loader, text="Validation")
    scheduler.step()

test(config, model, test_loader, text="Test")
