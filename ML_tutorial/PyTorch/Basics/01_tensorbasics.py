# coding=<utf-8> 
"""
(ref) https://youtu.be/x9JiIFvlUwk
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorbasics.py

* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'

"""

#%% 임포트 토치 
import torch 



# %% 01. 텐서 생성하기 
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3], [4,5,6], [7,8,9]] , 
                        dtype=torch.float32,
                        device=device,      # 텐서를 로드할 장치 설정
                        requires_grad=True  # for Autograd mode
                        )

print(my_tensor)
print(my_tensor.dtype)  # 텐서 객체의 attribute 출력 
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad) # Autograd 를 사용할 수 있다면 True 반환 


# %% 텐서를 초기화하는 다른 방법들 
x = torch.empty( size = (3,3) )  # 일반적으로는 그냥 값이 없는 3x3 텐서 (그런데 다 0으로 초기화되네?)
x = torch.zeros( (3, 3))
x = torch.rand( (3, 3) )  # unifor distribution 으로 0~1 사이의 값을 샘플링해서 만듬 
x = torch.ones( (3, 3)) 
x = torch.eye(3,3)   # identity tensor (I); 이건 사이즈를 튜플로 주지 않는다.
x = torch.arange(start=0 , end=5, step=1) # 0부터 5까지 1스텝으로 시퀀스 텐서를 만든다 
x = torch.linspace(start=0.1, end=1, steps=10) # 0.1 부터 1 까지 10개를 등간격으로 샘플링해서 만든다 
x = torch.empty(size=(1,5)).normal_(mean=0, std=1) # 정규분포로 샘플링 
x = torch.empty(size=(1,5)).uniform_(0, 1) # 0부터 1사이의 숫자를 uniform_distribution으로 샘플링 
x = torch.diag(torch.ones(3)) # 대각 행렬을 1로 채운다 
print(x)


# %% 초기화된 텐서의 자료형 바꾸기 (convert tensor types)
tensor = torch.arange(4)
print(tensor)
print(tensor.bool())   # boolen True/False 
print(tensor.short())  # Int16
print(tensor.long())   # Int64 (Important)
print(tensor.half())   # Float16 
print(tensor.float())  # Float32 (Important)
print(tensor.double()) # Flaot64




# %% 넘파이와 텐서 넘나들기(Array to Tensor conversion and vice-versa)


