# coding=<utf-8> 
"""
(ref) https://youtu.be/Jy4wM2X21u0
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/804c45e83b27c59defb12f0ea5117de30fe25289/ML/Pytorch/Basics/pytorch_simple_fullynet.py#L26-L35
(ref) https://jovian.ai/learn/deep-learning-with-pytorch-zero-to-gans/lesson/lesson-1-pytorch-basics-and-linear-regression

* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'

"""

"""
Fully-connected(FC) 네트워크를 구성하고 MNIST 데이터셋으로 학습하기 

1. Create a fully-connected network 

2. Set device 

3. Hyperparameters 

4. Load Data (MNIST)

5. Initialize network 

6. Loss and optimizer 

7. Train network 

8. Check accuracy on training & test to see how good our model
"""


#%% 임포트 토치 
import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F # 학습 안 되는 레이어들을 담은 패키지 ; # All functions that don't have any parameters, relu, tanh, etc. 
import torch.optim as optim  # 최적화 알고리즘을 담은 패키지 ; # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import DataLoader   # Gives easier dataset management and creates mini batches

import torchvision.datasets as datasets  # 이미지 데이터를 불러오고 변환하는 패키지 ;  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset




# ================================================================= #
#                1. Create a fully-connected network                #
# ================================================================= #
# %% 01. 심플한 FCnet 생성하기 
class NN(nn.Module): 
    def __init__(self, input_size: int, num_classes: int):
        super(NN, self).__init__()   # (ref) https://dojang.io/mod/page/view.php?id=2386
        """
        여기서는 학습이 가능한 계층을 설계한다. 
        예를 들어 nn 패키지를 활용하는 것들 

        (ex) nn.conv2d, nn.Linear, etc.
        """
        self.fc1 = nn.Linear(input_size, 50) 
        self.fc2 = nn.Linear(50, num_classes)



    def forward(self, x):   # 순전파 모듈 
        """
        여기서는 학습 파라미터가 없는 계층을 설계한다. 
        예를 들어 nn.functional 패키지를 활용한 것들 

        (ex) F.relu, F.max_pool2d, etc.
        """

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 



"""
모델 구조가 잘 만들어 졌는지 확인 

model = NN(784 , 10)        # MNIST 이미지 크기는 28x28 = 784, 클래스 개수 = 10
x = torch.randn(64, 784)    # Batch = 64, 특징벡터 길이 = 784
print(model(x).shape)

"""


# ================================================================= #
#                         2. Set device                             #
# ================================================================= #
# %% 02. 프로세스 장비 설정 
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')



# ================================================================= #
#                       3. Hyperparameters                          #
# ================================================================= #
# %% 03. 하이퍼파라미터 설정 
input_size = 784 
num_classes = 10 
learning_rate = 0.001 
batch_size = 64 
num_epochs = 1 


# ================================================================= #
#                      4.  Load Data (MNIST)                        #
# ================================================================= #
# %% 04. MNIST 데이터 로드 

"""
랜덤 발생 기준을 시드로 고정함. 
그러면 shuffle=True 이어도, 언제나 동일한 방식으로 섞여서 동일한 데이터셋을 얻을 수 있음. 
"""
torch.manual_seed(42)


train_dataset = datasets.MNIST( root='dataset/',    # 데이터가 위치할 경로 
                                train=True,         # train 용으로 가져오고 
                                transform=transforms.ToTensor(),  # 데이터 타입을 Tensor 형태로 변경 ; (ref) https://mjdeeplearning.tistory.com/81
                                download=True       # 해당 root에 데이터가 없으면 torchvision 으로 다운 받아라 
                                )

train_loader = DataLoader(  dataset=train_dataset,   # 로드 할 데이터 객체 
                            batch_size=batch_size,   # mini batch 덩어리 크기 설정 
                            shuffle=True             # 데이터 순서를 뒤섞어라 
                            )      


test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)                                               



# ================================================================= #
#                      5.  Initialize network                       #
# ================================================================= #
# %% 05. 모델 초기화
model = NN(input_size=input_size, num_classes=num_classes).to(device)



# ================================================================= #
#                      6.  Loss and optimizer                       #
# ================================================================= #
# %% 06. 손실 함수와 최적화 알고리즘 정의 
criterion = nn.CrossEntropyLoss()   
optimizer = optim.Adam( model.parameters(), lr=learning_rate, momentum=0.5 )  # 네트워크의 모든 파라미터를 전달한다 



# ================================================================= #
#                      7.  Train network                            #
# ================================================================= #
# %% 07. 학습 루프 

"""
# 학습하기 전에 모델이 AutoGrad를 사용해 학습할 수 있도록 train_mode 로 변환.

(1) backpropagation 계산이 가능한 상태가 됨.
(2) Convolution 또는 Linear 뿐만 아니라, 
    DropOut과 Batch Normalization 등의  파라미터를 가진 Layer들도 학습할 수 있는 상태가 된다. 
"""


for epoch in range(num_epochs):

    model.train()  

    for batch_idx, (data, targets) in enumerate(train_loader): # 미니배치 별로 iteration 
        # Get data to cuda if possible
        data = data.to(device=device)  # 미니 베치 데이터를 device 에 로드 
        targets = targets.to(device=device)  # 레이블 for supervised learning 
        
        # Get to correct shape
#        print(data.shape)    # torch.Size([64, 1, 28, 28])
        data = data.reshape(data.shape[0], -1)  # 네트워크 입력에 맞게 (28, 28) -> (784,) 로 펼치기(unroll)
                                                # torch.Size([64, 784])
        
        # forward
        scores = model(data)   # 모델이 예측한 값 
        loss = criterion(scores, targets)
        
        # backward
        optimizer.zero_grad()   # AutoGrad 하기 전에 매번 mini batch 별로 기울기 수치를 0으로 초기화 
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()



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

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
        
    num_correct = 0
    num_samples = 0

    model.eval()  
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')  # 소수 둘 째 자리 까지 표현


# %%
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
