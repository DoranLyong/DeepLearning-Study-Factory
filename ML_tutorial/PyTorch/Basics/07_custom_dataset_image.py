# coding=<utf-8> 
"""
(ref) https://youtu.be/ZoZHd0Zm3RY
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/aba36b89b438ca8f608a186f4d61d1b60c7f24e0/ML/Pytorch/Basics/custom_dataset/custom_dataset.py#L12-L29
(dataset) https://www.kaggle.com/dataset/c75fbba288ac0418f7786b16e713d2364a1a27936e63f4ec47502d73d6ef30ab


* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'
"""

"""
커스텀 데이터로 GoogLeNet 모델 학습하기 

1. Create CatsAndDogsDataset 

2. Set device 

3. Hyperparameters 

4. Load your custom dataset 

5. Initialize GoogLeNet network 

6. Loss and optimizer 

7. Train network and save the model 

8. Check accuracy on training & test to see how good our model
"""


#%% 임포트 라이브러리 
import os 
import os.path as osp

import pandas as pd  # .csv 형식의 데이터 프레임을 다루는 라이브러리 
from skimage import io  # 이미지 파일을 불러오기위한 라이브러리 
import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F # 학습 안 되는 레이어들을 담은 패키지 ; # All functions that don't have any parameters, relu, tanh, etc. 
import torch.optim as optim  # 최적화 알고리즘을 담은 패키지 ; # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import DataLoader   # Gives easier dataset management and creates mini batches
from torch.utils.data import Dataset    # 가져다쓸 데이터셋 객체를 지칭하는 클래스 (ref) https://huffon.github.io/2020/05/26/torch-data/
                                        
                                        
import torchvision
import torchvision.datasets as datasets  # 이미지 데이터를 불러오고 변환하는 패키지 ;  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset




# ================================================================= #
#                 1. Create CatsAndDogsDataset                      #
# ================================================================= #
# %% 01. 데이터셋 로더 생성하기 
"""
(dataset) https://www.kaggle.com/dataset/c75fbba288ac0418f7786b16e713d2364a1a27936e63f4ec47502d73d6ef30ab 
에서 데이터을 먼저 받는다. 

데이터셋은 ./dataset/archive 에 위치시킨다. 


# Dataset 클래스와 DataLoader 클래스의 관계:
- DataLoader와 Dataset 클래스의 상속관계 (ref) https://hulk89.github.io/pytorch/2019/09/30/pytorch_dataset/
- Dataset 클래스로 커스텀 데이터 셋을 만든다 => 생성된 데이터셋을 DataLoader 클래스로 전달해서 불러온다 (ref) https://wikidocs.net/57165
- (ref) https://doranlyong-ai.tistory.com/42
"""

class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        가져다쓸 데이터셋의 정보를 초기화한다. 
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        """
        초기화된 객체가 컨테이너 자료형을 가지고 있으면, 그것의 길이를 반환한다

        __len__() 매직 함수를 사용하면 내장 함수 len()을 사용할 수 있다 

        (ref) https://dgkim5360.tistory.com/entry/python-duck-typing-and-protocols-why-is-len-built-in-function
        (ref) https://kwonkyo.tistory.com/234
        (ref) https://medium.com/humanscape-tech/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9D%98-%EC%8A%A4%ED%8E%98%EC%85%9C-%EB%A9%94%EC%84%9C%EB%93%9C-special-method-2aea6bc4f2b9

        """
        return len(self.annotations) # 로드된 데이터의 개수(길이) 를 반환한다 


    def __getitem__(self, index):
        """
        데이터셋 시퀀스에서 특정 index에 해당하는 아이템을 가져온다 (= 객체에 indexing 기능을 사용할 수 있음). 

        (ref) http://hyeonjae-blog.logdown.com/posts/776615-python-getitem-len
        """
        img_path = osp.join(self.root_dir, self.annotations.iloc[index, 0]) # 행번호 index 에서 열번호 0 에 해당하는 아이템을 가져온다. (ex) cat.1.jpg
                                                                            # (ref) https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
                                                                            # (ref) https://devpouch.tistory.com/47
        image = io.imread(img_path) # 해당 경로에서 이미지 파일을 불러온다 
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))  # 타겟 레이블 정보를 가져온다 


        if self.transform:
            image = self.transform(image)

        return (image, y_label)                                                                                                            









        

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
in_channel = 1
num_classes = 10 
learning_rate = 0.001
batch_size = 64
num_epochs = 10

load_model = False  # 체크포인트 모델을 가져오려면 True 


# ================================================================= #
#                      4.  Load Data your dataset                   #
# ================================================================= #
# %% 04. 커스텀 데이터 로드 

"""
랜덤 발생 기준을 시드로 고정함. 
그러면 shuffle=True 이어도, 언제나 동일한 방식으로 섞여서 동일한 데이터셋을 얻을 수 있음. 
"""
torch.manual_seed(42)


dataset = CatsAndDogsDataset(   csv_file= osp.join('dataset', 'archive', 'cats_dogs.csv'), 
                                root_dir = osp.join('dataset', 'archive', 'cats_dogs_resized'), 
                                transform = transforms.ToTensor(),  # 데이터 타입을 Tensor 형태로 변경 ; (ref) https://mjdeeplearning.tistory.com/81
                                )


"""
로드된 데이터셋 쪼개기

# Dataset is actually a lot larger ~25k images, just took out 10 pictures
# to upload to Github. It's enough to understand the structure and scale
# if you got more images.

이번 예제에서는 dataset 객체로부터 train:test = 5개 : 5개 로 쪼갠다 (현재 주어진 데이터는 총 10개). 
"""

train_set, test_set = torch.utils.data.random_split(dataset, [5, 5])

train_loader = DataLoader(  dataset=train_set,       # 로드 할 데이터 객체 
                            batch_size=batch_size,   # mini batch 덩어리 크기 설정 
                            shuffle=True             # 데이터 순서를 뒤섞어라 
                            )   

test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)



"""
torchvision의 datasets 패키지를 사용했을 때와는 어떻게 다른지 비교해보자. 
"""


# ================================================================= #
#                      5.  Initialize network                       #
# ================================================================= #
# %% 05. 모델 초기화
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

# ================================================================= #
#                      9. Checkpoint save & load                    #
# ================================================================= #
#%% 09. 체크포인트를 저장하고 다시 로드하기 
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



# ================================================================= #
#                  6.  Loss and optimizer  & load checkpoint        #
# ================================================================= #
# %% 06. 손실 함수와 최적화 알고리즘 정의 
criterion = nn.CrossEntropyLoss()   
optimizer = optim.Adam( model.parameters(), lr=learning_rate)  # 네트워크의 모든 파라미터를 전달한다 

if load_model:
    try: 
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    except OSError as e: 
        print(e)
        pass 


# ================================================================= #
#                      7.  Train network & save                     #
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

    losses = [] 

    
    if epoch % 3 == 0: # 3주기 마다 모델 저장  
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()} # 체크 포인트 상태 
        # Try save checkpoint
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader): # 미니배치 별로 iteration 
        # Get data to cuda if possible
        data = data.to(device=device)  # 미니 베치 데이터를 device 에 로드 
        targets = targets.to(device=device)  # 레이블 for supervised learning 
        
        
        # forward
        scores = model(data)   # 모델이 예측한 수치 
        loss = criterion(scores, targets)
        losses.append(loss.item())
        
        # backward
        optimizer.zero_grad()   # AutoGrad 하기 전에 매번 mini batch 별로 기울기 수치를 0으로 초기화 
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()
    
    mean_loss = sum(losses) / len(losses)
    print(f"Loss at epoch {epoch} was {mean_loss:.5f}") # 소수 다섯째 자리까지 표시 



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
    num_correct = 0
    num_samples = 0

    model.eval()  
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')  # 소수 둘 째 자리 까지 표현


# %%
print("Checking accuracy on training data")
check_accuracy(train_loader, model)

print("Checking accuracy on test data")
check_accuracy(test_loader, model)

# %%
