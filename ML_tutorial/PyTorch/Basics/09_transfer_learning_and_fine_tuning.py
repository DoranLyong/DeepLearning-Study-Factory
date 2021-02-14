#-*- coding: utf-8 -*-

"""
(ref) https://youtu.be/qaDe0qQZ5AQ
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/804c45e83b27c59defb12f0ea5117de30fe25289/ML/Pytorch/Basics/pytorch_pretrain_finetune.py#L33-L54
(ref) https://github.com/DoranLyong/ResNet-tutorial/blob/main/ResNet_pytorch/ResNet18_for_CIFAR-10.py

* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""

"""
사전 학습된 VGG16 모델을 불러와서 CIFAR-10 클래스 분류에 맞게 fine-tuning 하기 
    
1. Load pretrained VGG16 

2. Set device 

3. Hyperparameters 

4. Load Data (CIFAR10)

5. Initialize & Finetune the network 

6. Loss and optimizer 

7. Train network 

8. Check accuracy on training & test to see how good our model
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

import torchvision.datasets as datasets  # 이미지 데이터를 불러오고 변환하는 패키지 ;  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision



# ================================================================= #
#                     1. Load pretrained VGG16                     #
# ================================================================= #
# %% 01.  사전 학습된 VGG16 불러오기 

""" Simple Identity class that let's input pass without changes
"""
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


""" Load the pretrain model 
"""
model = torchvision.models.vgg16(pretrained=True)
print(model)



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


# ================================================================= #
#                      4.  Load Data (CIFAR10)                      #
# ================================================================= #
# %% 04. CIFAR10 데이터 로드 
"""
랜덤 발생 기준을 시드로 고정함. 
그러면 shuffle=True 이어도, 언제나 동일한 방식으로 섞여서 동일한 데이터셋을 얻을 수 있음. 
"""
torch.manual_seed(42)


transform_train = transforms.Compose([  transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), # 데이터 타입을 Tensor 형태로 변경 ; (ref) https://mjdeeplearning.tistory.com/81
                                    ])

transform_test = transforms.Compose([   transforms.ToTensor(),
                                    ])



train_dataset = datasets.CIFAR10( root='dataset/',    # 데이터가 위치할 경로 
                                train=True,         # train 용으로 가져오고 
                                transform=transform_train,  
                                download=True       # 해당 root에 데이터가 없으면 torchvision 으로 다운 받아라 
                                )

train_loader = DataLoader(  dataset=train_dataset,   # 로드 할 데이터 객체 
                            batch_size=batch_size,   # mini batch 덩어리 크기 설정 
                            shuffle=True,            # 데이터 순서를 뒤섞어라 
                            num_workers=4,           # Broken Pipe 에러 뜨면 지우기 
                            )      


test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transform_test, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4) 



# ================================================================= #
#            5.  Initialize & Finetune the network                  #
# ================================================================= #
# %% 05. 모델 초기화 및 fine-tuning 

"""
If you want to do finetuning then set requires_grad = False
Remove these two lines if you want to train entire model,
and only want to load the pretrain weights.
"""
for param in model.parameters():
    param.requires_grad = False   # Transfer learning 을 한다면 True (= 이 블록은 삭제해도 됨)
                                  # Finetune 만 하겠다면 False => backbone 은 학습할 필요 없으니까


""" Finetuning
"""
model.avgpool = Identity() # VGG16 의 (avgpool) 계층을 Identity mapping 으로 바꿈  


model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, num_classes)   # CIFAR10 분류의 클래스 개수 만큼 Finetune
                                 )   

print(model) # (avgpool) 와 (classifier) 부분이 어떻게 바뀌었는지 확인 

"""
모델 구조가 잘 만들어 졌는지 확인 

x = torch.randn(64, 3, 32, 32)    # Batch = 64, 이미지 크기= 3 x 28 x 28
print(model(x).shape)   # torch.Size([64, 10])
"""                                 

#%%
""" 모델을 device 에 로드 
"""
model = model.to(device=device)      
model = torch.nn.DataParallel(model)# 데이터 병렬처리          # (ref) https://tutorials.pytorch.kr/beginner/blitz/data_parallel_tutorial.html
                                    # 속도가 더 빨라지진 않음   # (ref) https://tutorials.pytorch.kr/beginner/former_torchies/parallelism_tutorial.html
                                    # 오히려 느려질 수 있음    # (ref) https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b
cudnn.benchmark = False
                     


# ================================================================= #
#                  6.  Loss and optimizer  & load checkpoint        #
# ================================================================= #
# %% 06. 손실 함수와 최적화 알고리즘 정의 
criterion = nn.CrossEntropyLoss()   
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 



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
# %% Train block
def train(epoch):
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

        _, predictions = scores.max(1)
        num_correct += predictions.eq(targets).sum().item()   #  맞춘 샘플 개수 
        num_samples += predictions.size(0)   # 총 예측한 샘플 개수 (맞춘 것 + 틀린 것)

        # backward
        optimizer.zero_grad()   # AutoGrad 하기 전에 매번 mini batch 별로 기울기 수치를 0으로 초기화 
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()




        if batch_idx % batch_size == 0:  
            print(f"\n@batch: {str(batch_idx)}")
            print(f"train acc: {str((predictions == targets).sum().item() / predictions.size(0) )}")
            print(f"train loss: {loss.item()}")

    print(f"\nTotal train acc: {float(num_correct)/float(num_samples)*100:.2f}")

    mean_loss = sum(train_losses) / len(train_losses)
    print(f"Mean loss of train: {mean_loss:.5f}") # 소수 다섯째 자리까지 표시 



# ================================================================= #
# 8.  Check accuracy on training & test to see how good our model   #
# ================================================================= #
# %% 08. 학습 정확도 확인
"""
(1) 평가 단계에서는 모델에 evaluation_mode 를 설정한다 
    - 학습 가능한 파라미터가 있던 계층들을 잠금 
(2) AutoGrad engine 을 끈다 ; torch.no_grad() 
    - backpropagation 이나 gradient 계산 등을 꺼서 memory usage를 줄이고 속도를 높임.
    
    - (ref) http://taewan.kim/trans/pytorch/tutorial/blits/02_autograd/
"""

# %% Validation block
def test(epoch):
    print(f"\n*****[ Validation epoch: {epoch} ]*****")
    model.eval()  


    val_losses = [] 
    num_correct = 0 
    num_samples = 0     

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader): # 미니배치 별로 iteration 
            # Get data to cuda if possible
            data = data.to(device=device)  # 미니 베치 데이터를 device 에 로드 
            targets = targets.to(device=device)  # 레이블 for supervised learning 

            # forward
            scores = model(data)   # 모델이 예측한 수치 
            loss = criterion(scores, targets)
            val_losses.append(loss.item())

            _, predictions = scores.max(1)
            num_correct += predictions.eq(targets).sum().item()   #  맞춘 샘플 개수 
            num_samples += predictions.size(0)   # 총 예측한 샘플 개수 (맞춘 것 + 틀린 것)


        print(f"\nValidation acc: {float(num_correct)/float(num_samples)*100:.2f}")

        mean_loss = sum(val_losses) / len(val_losses)
        print(f"Mean loss of validation: {mean_loss:.5f}") # 소수 다섯째 자리까지 표시 



# ================================================================= #
#                         Train & Validation 루프                   #
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
    test(epoch)
# %%        