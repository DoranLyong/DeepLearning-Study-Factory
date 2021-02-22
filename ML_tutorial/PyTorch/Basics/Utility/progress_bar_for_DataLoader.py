#-*- coding: utf-8 -*-

"""
(ref) https://youtu.be/RKHopFfbPao
(ref) https://aladdinpersson.medium.com/how-to-get-a-progress-bar-in-pytorch-72bdbf19b35c
(ref) https://github.com/DoranLyong/ResNet-tutorial/blob/main/ResNet_pytorch/ResNet18_for_CIFAR-10.py

* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""

"""
쌈박하게 활용할 수 있는 Train-loop 3종 템플릿 

(방법1) 단순함 

(방법2) batch_idx 추가 

(방법3) progress_bar 콘솔에다가 기타 loss, acc 정보 추가 

"""


#%% 임포트 토치 
import os.path as osp
import os

from tqdm import tqdm 
import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F # 학습 안 되는 레이어들을 담은 패키지 ; # All functions that don't have any parameters, relu, tanh, etc. 
import torch.backends.cudnn as cudnn    # https://hoya012.github.io/blog/reproducible_pytorch/
                                        # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
import torch.optim as optim  # 최적화 알고리즘을 담은 패키지 ; # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import DataLoader   # Gives easier dataset management and creates mini batches

import torchvision.datasets as datasets  # 이미지 데이터를 불러오고 변환하는 패키지 ;  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset





# ================================================================= #
#                            Create a CNN                           #
# ================================================================= #
# %% 01. 심플한 CNN 생성하기 
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        """
        여기서는 학습이 가능한 계층을 설계한다. 
        예를 들어 nn 패키지를 활용하는 것들 
        (ex) nn.conv2d, nn.Linear, etc.
        """

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    
    def forward(self, x):
        """
        여기서는 학습 파라미터가 없는 계층을 설계한다. 
        예를 들어 nn.functional 패키지를 활용한 것들 
        (ex) F.relu, F.max_pool2d, etc.
        """
                
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # 펼치기 (unroll)
        x = self.fc1(x)
        
        return x


# ================================================================= #
#                            Set device                             #
# ================================================================= #
# %% 02. 프로세스 장비 설정 
gpu_no = 0  # gpu_number 
device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')




# ================================================================= #
#                           Load Data (MNIST)                       #
# ================================================================= #
# %% 04. MNIST 데이터 로드 
"""
랜덤 발생 기준을 시드로 고정함. 
그러면 shuffle=True 이어도, 언제나 동일한 방식으로 섞여서 동일한 데이터셋을 얻을 수 있음. 
"""
torch.manual_seed(42)


transform_train = transforms.Compose([ #Compose makes it possible to have many transforms
                                        transforms.ToTensor(), # 데이터 타입을 Tensor 형태로 변경 ; (ref) https://mjdeeplearning.tistory.com/81 # Finally converts PIL image to tensor so we can train w. pytorch
                                    ])                                    


train_dataset = datasets.MNIST( root='dataset/',    # 데이터가 위치할 경로 
                                train=True,         # train 용으로 가져오고 
                                transform=transform_train,  
                                download=True       # 해당 root에 데이터가 없으면 torchvision 으로 다운 받아라 
                                )

train_loader = DataLoader(  dataset=train_dataset,   # 로드 할 데이터 객체 
                            batch_size=64,           # mini batch 덩어리 크기 설정 
                            shuffle=True,            # 데이터 순서를 뒤섞어라 
                            num_workers=4,          # Broken Pipe 에러 뜨면 지우기 
                            )      



# ================================================================= #
#                        Initialize network                         #
# ================================================================= #
# %% 모델 초기화
model = CNN().to(device)
model = torch.nn.DataParallel(model)# 데이터 병렬처리          # (ref) https://tutorials.pytorch.kr/beginner/blitz/data_parallel_tutorial.html
                                    # 속도가 더 빨라지진 않음   # (ref) https://tutorials.pytorch.kr/beginner/former_torchies/parallelism_tutorial.html
                                    # 오히려 느려질 수 있음    # (ref) https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b
cudnn.benchmark = True


# ================================================================= #
#                   Loss and optimizer  & load checkpoint           #
# ================================================================= #
# %%  손실 함수와 최적화 알고리즘 정의 
criterion = nn.CrossEntropyLoss()   
optimizer = optim.SGD( model.parameters(), lr=1e-2,  momentum=0.9, weight_decay=0.0002)  # 네트워크의 모든 파라미터를 전달한다 


# ================================================================= #
#                            Train-loop                             #
# ================================================================= #

NUM_EPOCHS = 100

# %% (방법1)
for epoch in range(NUM_EPOCHS):    
    for data, targets in tqdm(train_loader, leave=False):
        """
        * leave=False ; make-up 된 progress_bar 는 지우고 다시 시작         
        """
        pass 



# %% (방법2)
for epoch in range(NUM_EPOCHS):    
    for batch_idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        """
        * total=len(train_loader) ;  순회할 데이터의 총 길이 
        * leave=False ; make-up 된 progress_bar 는 지우고 다시 시작 
        """
        pass 



# %% (방법3) - 쌈박한 방법 
for epoch in range(NUM_EPOCHS):    
    """
    (방법2)의 순회 객체를 루프 밖에서 인스턴스화
    """
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    
    model.train()


    for batch_idx, (data, targets) in loop:
        # Get data to cuda if possible
        data = data.to(device=device)  # 미니 베치 데이터를 device 에 로드 
        targets = targets.to(device=device)  # 레이블 for supervised learning 


        # forward 
        scores = model(data)   # 모델이 예측한 수치
        loss = criterion(scores, targets)


        # backward
        optimizer.zero_grad()   # AutoGrad 하기 전에 매번 mini batch 별로 기울기 수치를 0으로 초기화 
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()


        # Update progress bar 
        """
        이거 알아 두면 쌈박하게 콘솔을 이용할 수 있음.

        여기 acc=torch.rand(1).item() 부분은 임의로 넣었음. 
        자신의 task에 맞게 accuracy 를 계산해서 넣으면 됨. 
        """
        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix( acc=torch.rand(1).item(), loss=loss.item())

# %%
