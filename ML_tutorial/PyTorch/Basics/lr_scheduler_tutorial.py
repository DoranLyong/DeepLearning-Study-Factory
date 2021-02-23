#-*- coding: utf-8 -*-

"""
(ref) https://youtu.be/P31hB37g4Ak
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/804c45e83b27c59defb12f0ea5117de30fe25289/ML/Pytorch/Basics/pytorch_lr_ratescheduler.py#L45-L78
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
import torchvision





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
learning_rate = 1e-2
batch_size = 128
NUM_EPOCHS = 40

SEED = 42 # set seed 


# ================================================================= #
#                         4.Load Data (CIFAR10)                     #
# ================================================================= #
# %% 04. CIFAR10 데이터 로드 
"""
랜덤 발생 기준을 시드로 고정함. 
그러면 shuffle=True 이어도, 언제나 동일한 방식으로 섞여서 동일한 데이터셋을 얻을 수 있음. 
"""

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



transform_train = transforms.Compose([ #Compose makes it possible to have many transforms
                                        transforms.ToTensor(), # 데이터 타입을 Tensor 형태로 변경 ; (ref) https://mjdeeplearning.tistory.com/81 # Finally converts PIL image to tensor so we can train w. pytorch
                                    ])                                    


train_dataset = datasets.CIFAR10( root='dataset/',    # 데이터가 위치할 경로 
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
#                       5. Initialize network                         #
# ================================================================= #
# %% 05. 모델 초기화

model = torchvision.models.googlenet(pretrained=True)

for param in model.parameters():
    """
    (ref) https://github.com/DoranLyong/DeepLearning_model_factory/blob/master/ML_tutorial/PyTorch/Basics/09_transfer_learning_and_fine_tuning.py
    """ 
    param.requires_grad = False # Transfer learning 을 한다면 True (= 이 블록은 삭제해도 됨)
                                # Finetune 만 하겠다면 False => backbone 은 학습할 필요 없으니까

model.fc = nn.Linear(1024, num_classes)                                


model = model.to(device=device)
model = torch.nn.DataParallel(model)# 데이터 병렬처리          # (ref) https://tutorials.pytorch.kr/beginner/blitz/data_parallel_tutorial.html
                                    # 속도가 더 빨라지진 않음   # (ref) https://tutorials.pytorch.kr/beginner/former_torchies/parallelism_tutorial.html
                                    # 오히려 느려질 수 있음    # (ref) https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b
cudnn.benchmark = True                            




# ================================================================= #
#                       6.  Loss and optimizer                      #
# ================================================================= #
# %% 06. 손실 함수와 최적화 알고리즘 정의 
criterion = nn.CrossEntropyLoss()   
optimizer = optim.SGD( model.parameters(), lr=learning_rate,  momentum=0.9, weight_decay=0.0002)  # 네트워크의 모든 파라미터를 전달한다 


# ================================================================= #
#                       6-1.  Define Scheduler                      #
# ================================================================= #
# %% 스케쥴러 정의 

# Define Scheduler
"""
(ref) https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
"""
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1, patience=5, verbose=True)



# ================================================================= #
#                              Train-loop                           #
# ================================================================= #

# %% Train-loo  정의 
def train(epoch):
    global NUM_EPOCHS, scheduler 

    model.train()  

    train_losses = [] 
    num_correct = 0 
    num_samples = 0 

    # (ref) https://github.com/DoranLyong/DeepLearning_model_factory/blob/master/ML_tutorial/PyTorch/Basics/Utility/progress_bar_for_DataLoader.py
    loop = tqdm(enumerate(train_loader), total=len(train_loader))  
    

    for batch_idx, (data, targets) in loop: # 미니배치 별로 iteration 
        """Get data to cuda if possible
        """
        data = data.to(device=device)  # 미니 베치 데이터를 device 에 로드 
        targets = targets.to(device=device)  # 레이블 for supervised learning 

        """forward
        """
        scores = model(data)   # 모델이 예측한 수치 
        loss = criterion(scores, targets)
        train_losses.append(loss.item())

        _, predictions = scores.max(1)
        num_correct += predictions.eq(targets).sum().item()   #  맞춘 샘플 개수 
        num_samples += predictions.size(0)   # 총 예측한 샘플 개수 (맞춘 것 + 틀린 것)

        """backward
        """
        optimizer.zero_grad()   # AutoGrad 하기 전에(=역전파 실행전에) 매번 mini batch 별로 기울기 수치를 0으로 초기화 
        loss.backward()         # (ref) https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html
        
        """gradient descent or adam step
        """
        optimizer.step()
        

        """ progress bar with tqdm
        """
        # (ref) https://github.com/DoranLyong/DeepLearning_model_factory/blob/master/ML_tutorial/PyTorch/Basics/Utility/progress_bar_for_DataLoader.py
        # (ref) https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/
        # LR 수치 가져와 보기 ; (ref) https://discuss.pytorch.org/t/how-to-retrieve-learning-rate-from-reducelronplateau-scheduler/54234

        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}], LR={ optimizer.param_groups[0]['lr'] :.1e}")
        

        if batch_idx % batch_size == 0:  
            loop.set_postfix( acc=(predictions == targets).sum().item() / predictions.size(0), loss=loss.item(),  batch=batch_idx)


    total_acc = float(num_correct)/float(num_samples)*100
    mean_loss = sum(train_losses) / len(train_losses)

    print(f"\nTotal train acc: {total_acc:.2f}%")
    print(f"Mean loss of train: {mean_loss:.5f}") # 소수 다섯째 자리까지 표시 



    """After each epoch do scheduler.step, 
        note in this scheduler we need to send in loss for that epoch!

        lr 수치가 바뀌면 뭐가 뜨나? 
    """
    scheduler.step(mean_loss) # Decay Learning Rate
    


# %% Train-loop 실행 
for epoch in range(1, NUM_EPOCHS+1):
    train(epoch)
    

