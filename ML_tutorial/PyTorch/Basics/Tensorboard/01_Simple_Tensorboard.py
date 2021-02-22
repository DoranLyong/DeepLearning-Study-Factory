#-*- coding: utf-8 -*-

"""
(ref) https://youtu.be/RLqsxWaQdHE
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/79f2e1928906f3cccbae6c024f3f79fd05262cd1/ML/Pytorch/Basics/pytorch_tensorboard_.py
(ref) https://github.com/DoranLyong/ResNet-tutorial/blob/main/ResNet_pytorch/ResNet18_for_CIFAR-10.py

* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""

"""
CNN 네트워크를 구성하고 MNIST 데이터셋으로 학습하기  & 학습 로그를 TensorBoard 로 표현하기 
    
1. Create a CNN

2. Set device 

3. Hyperparameters 

4. Load Data (MNIST)

5. Initialize & Finetune the network 

6. Loss and optimizer 

7. Train network 

"""

#%% 임포트 토치 
import os.path as osp
import os

import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F # 학습 안 되는 레이어들을 담은 패키지 ; # All functions that don't have any parameters, relu, tanh, etc. 
import torch.backends.cudnn as cudnn    # https://hoya012.github.io/blog/reproducible_pytorch/
                                        # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
import torch.optim as optim  # 최적화 알고리즘을 담은 패키지 ; # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import DataLoader   # Gives easier dataset management and creates mini batches
from torch.utils.tensorboard import SummaryWriter # 텐서보드 연동 # to print to tensorboard

import torchvision.datasets as datasets  # 이미지 데이터를 불러오고 변환하는 패키지 ;  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset




# ================================================================= #
#                           1. Create a CNN                         #
# ================================================================= #
# %% 01. 심플한 CNN 생성하기 
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()  # (ref) https://dojang.io/mod/page/view.php?id=2386

        """
        여기서는 학습이 가능한 계층을 설계한다. 
        예를 들어 nn 패키지를 활용하는 것들 

        (ex) nn.conv2d, nn.Linear, etc.
        """

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
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
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


"""
모델 구조가 잘 만들어 졌는지 확인 

model = CNN()        
x = torch.randn(64, 1, 28, 28)    # Batch = 64, 이미지 크기= 28 x 28
print(model(x).shape)   # torch.Size([64, 10])
"""



# ================================================================= #
#                         2. Set device                             #
# ================================================================= #
# %% 02. 프로세스 장비 설정 
gpu_no = 0  # gpu_number 
device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')



# ================================================================= #
#                       3. Hyperparameters                          #
# ================================================================= #
# %% 03. 하이퍼파라미터 설정 
num_classes = 10 
learning_rate =  1e-3
batch_size = 64
num_epochs = 5

in_channels = 1


# ================================================================= #
#                        4.  Load Data (MNIST)                      #
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
                            batch_size=batch_size,   # mini batch 덩어리 크기 설정 
                            shuffle=True,            # 데이터 순서를 뒤섞어라 
                            num_workers=4,          # Broken Pipe 에러 뜨면 지우기 
                            ) 



# ================================================================= #
#            5.  Initialize & Finetune the network                  #
# ================================================================= #
# %% 05. 모델 초기화 및 fine-tuning 
model = CNN(in_channels=in_channels, num_classes=num_classes)
model.to(device)

model = torch.nn.DataParallel(model)# 데이터 병렬처리          # (ref) https://tutorials.pytorch.kr/beginner/blitz/data_parallel_tutorial.html
                                    # 속도가 더 빨라지진 않음   # (ref) https://tutorials.pytorch.kr/beginner/former_torchies/parallelism_tutorial.html
                                    # 오히려 느려질 수 있음    # (ref) https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b
cudnn.benchmark = False



# ================================================================= #
#           6.  Loss and optimizer  & TensorBoard                   #
# ================================================================= #
# %% 06. 손실 함수와 최적화 알고리즘 정의 
criterion = nn.CrossEntropyLoss()   
optimizer = optim.Adam( model.parameters(), lr=learning_rate)  # 네트워크의 모든 파라미터를 전달한다 


""" 텐서보드 객체 초기화 
"""
writer = SummaryWriter(f'runs/MNIST/tryingout_tensorboard') # 학습 로그(log)가 저장되는 경로 초기화 




# ================================================================= #
#                      7.  Train network                            #
# ================================================================= #
# %% 07. 학습 블록 

"""
# 학습하기 전에 모델이 AutoGrad를 사용해 학습할 수 있도록 train_mode 로 변환.
(1) backpropagation 계산이 가능한 상태가 됨.
(2) Convolution 또는 Linear 뿐만 아니라, 
    DropOut과 Batch Normalization 등의  파라미터를 가진 Layer들도 학습할 수 있는 상태가 된다. 
"""



step = 0


# %% Train block
def train(epoch):
    global step

    print(f"\n*****[ Train epoch: {epoch} ]*****")
    model.train()  

    train_losses = [] 
    num_correct = 0 
    num_samples = 0 

    for batch_idx, (data, targets) in enumerate(train_loader): # 미니배치 별로 iteration 
        # Get data to cuda if possible
        data = data.to(device=device)  # 미니 베치 데이터를 device 에 로드 
        targets = targets.to(device=device)  # 레이블 for supervised learning 

        # forward
        scores = model(data)   # 모델이 예측한 수치 
        loss = criterion(scores, targets)
        train_losses.append(loss.item())

        # Calculate 'running' training accuracy
        _, predictions = scores.max(1)
        num_correct += predictions.eq(targets).sum().item()   #  맞춘 샘플 개수 
        num_samples += predictions.size(0)   # 총 예측한 샘플 개수 (맞춘 것 + 틀린 것)
        
        running_train_acc = float((predictions == targets).sum()) / float(predictions.size(0)) # 매번 mini-batch 마다의 정확도

        # backward
        optimizer.zero_grad()   # AutoGrad 하기 전에 매번 mini batch 별로 기울기 수치를 0으로 초기화 
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()



        """ 텐서보드 로그 
        """
        writer.add_scalar('Training loss', loss, global_step=step)
        writer.add_scalar('Training Accuracy', running_train_acc, global_step=step)

        step += 1


        if batch_idx % batch_size == 0:  
            print(f"\n@batch: {str(batch_idx)}")
            print(f"train acc: {str((predictions == targets).sum().item() / predictions.size(0) )}")
            print(f"train loss: {loss.item()}")

    print(f"\nTotal train acc: {float(num_correct)/float(num_samples)*100:.2f}")

    mean_loss = sum(train_losses) / len(train_losses)
    print(f"Mean loss of train: {mean_loss:.5f}") # 소수 다섯째 자리까지 표시 



# ================================================================= #
#                            Train 루프                              #
# ================================================================= #
# %% 일정 에폭마다 학습률 줄이기 
def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 5:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# %% 훈련 루프 실행 
for epoch in range(0, num_epochs):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    