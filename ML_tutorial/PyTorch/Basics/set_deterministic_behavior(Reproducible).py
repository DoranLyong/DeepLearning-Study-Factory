#-*- coding: utf-8 -*-

"""
(ref) https://youtu.be/1SZocGaCAr8
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/set_deterministic_behavior/pytorch_set_seeds.py
(ref) https://github.com/DoranLyong/ResNet-tutorial/blob/main/ResNet_pytorch/ResNet18_for_CIFAR-10.py

* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""

"""
결과가 재구현이 가능하도록 (매번 새롭게 실행해도 결과는 동일하게) 하이퍼파라미터 설정하기 

1. Reproducible Results (재구현이 가능하게 끔, random number에 seed 할당하기 )
    - seed를 설정하면 매번 실행 할 때 마다 동일한 난수로 시작한다 

2. CPU 일 때,  CUDA 일 때 .
"""


#%% 임포트 토치 
import os
import random

import numpy as np 
import torch 
import torch.backends.cudnn as cudnn    # https://hoya012.github.io/blog/reproducible_pytorch/
                                        # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not


seed = 42 # set seed 

os.environ['PYTHONHASHSEED'] = str(seed)   # 파이썬 시드 고정 ; (ref) https://eda-ai-lab.tistory.com/535

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# %%
x = torch.rand((5, 5))
print(torch.einsum("ii-> ", x))   # (ref) https://youtu.be/pkVwUVEHmfI


# %% if using cuda 
"""
(ref) https://hoya012.github.io/blog/reproducible_pytorch/
"""


torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)  # if using multi-GPU 
cudnn.deterministic = True   # (ref) https://stackoverflow.com/questions/56354461/reproducibility-and-performance-in-pytorch
cudnn.benchmark = False 
