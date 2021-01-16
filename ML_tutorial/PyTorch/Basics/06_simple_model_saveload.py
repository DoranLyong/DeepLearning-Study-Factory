# coding=<utf-8> 
"""
(ref) https://youtu.be/g6kQl_EFn84
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/804c45e83b27c59defb12f0ea5117de30fe25289/ML/Pytorch/Basics/pytorch_loadsave.py#L26-L34


* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'
"""

"""
학습된 모델을 저장하고 checkpoint에서 저장된 모델을 다시 로드하기.

1. save the model as you train

2. , and then load before continuining training at another point

"""


#%% 임포트 토치 
import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F # 학습 안 되는 레이어들을 담은 패키지 ; # All functions that don't have any parameters, relu, tanh, etc. 
import torch.optim as optim  # 최적화 알고리즘을 담은 패키지 ; # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import DataLoader   # Gives easier dataset management and creates mini batches

import torchvision
import torchvision.datasets as datasets  # 이미지 데이터를 불러오고 변환하는 패키지 ;  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset



# ================================================================= #
#                         1. Save checkpoint                        #
# ================================================================= #
# %% 01. 체크 포인트에서 저장하기 
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# ================================================================= #
#                         2. Load checkpoint                        #
# ================================================================= #    
# %% 02. 체크 포인트 모델 불러오기
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# ================================================================= #
#                         3. Template practice                      #
# ================================================================= #      
# %% 03. 활용방법 
# Initialize network
model = torchvision.models.vgg16(pretrained=False)
optimizer = optim.Adam(model.parameters())

checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}

# Try save checkpoint
save_checkpoint(checkpoint)

# Try load checkpoint
load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

