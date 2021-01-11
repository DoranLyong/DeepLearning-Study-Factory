# coding=<utf-8> 
"""
(ref) https://youtu.be/x9JiIFvlUwk
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorbasics.py

* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'

"""

#%% 임포트 토치 
import torch 


# ================================================================= #
#                        Initializing Tensor                        #
# ================================================================= #
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
import numpy as np 
np_array = np.zeros( (5,5)) 
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy() 



# =============================================================================== #
#                        Tensor Math & Comparison Operations                      #
# =============================================================================== #
# %% 02. 텐서 연산 (Math & Comparison Operations)
x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])


#%% 덧셈(addition)
z1 = torch.empty(3, 3)
torch.add(x, y, out=z1)  

z2 = torch.add(x, y)

z3 = x + y    # 모두 같은 결과 


# %% 뺄셈(subtraction)
z = x - y 


# %% 나눗셈(division)
z = torch.true_divide(x, y)


# %% 인플레이스 연산(inplace operations)  (ref) https://discuss.pytorch.org/t/what-is-in-place-operation/16244
"""
연산 후 결과를 새로운 객체에 복사해서 할 당하지 않고 바로 바꾸기 떄문에 
메모리 관리에 좋다. 
"""
t = torch.ones(3)
t.add_(x)  # in-place 연산은 항상 뒤에 _ 기호가 붙음 
 
t += x  # 이것은 in-place 연산 
t = t + x  # 이건 in-place 연산이 아님 


# %% 지수 연산(exponentiation)
z = x.pow(2)
z = x ** 2     

# %% 간단한 비교연산(simple comparion)
z = x > 0 
print(z)

z = x < 0 
print(z)

# %% 행렬 곱셈(matrix multiplication)
x1 = torch.rand( (2,5) )
x2 = torch.rand( (5,3) )

x3 = torch.mm(x1, x2)  #  
x3 = x1.mm(x2) 

# %% 행렬 멱연산(matrix exponentiation) 
matrix_exp = torch.ones((2, 2))
print(matrix_exp)

print(matrix_exp.matrix_power(3)) # matrix_exp * matrix_exp * matrix_exp  행렬 곱셈 3번 


# %% 요소별 곱셈 (element-wise mult.)
z = x * y 
print(z)

# %% 내적 (dot product)
z = torch.dot(x, y)
print(z) 

# %% 배치 행렬 곱 (Batch Matrix Multiplication)
batch = 32 
n = 10 
m = 20 
p = 30 

tensor1 = torch.rand( (batch, n, m) ) 
tensor2 = torch.rand( (batch, m, p) )  
out_bmm = torch.bmm( tensor1, tensor2 ) # (batch x n x m ) * (batch x m x p) = (batch x n x p)


# %% 브로드캐스팅 (broadcasting)
x1 = torch.rand( (5, 5))
x2 = torch.rand( (1, 5)) 

z = x1 - x2 
z = x1 ** x2
print(z)  # (5,5) shape

# %% 기타 유용한 연산 (other useful tensor operation)
sum_x = torch.sum(x, dim=0)  # 0번째 축 방향으로 더해라 

values, indices = torch.max(x, dim=0) # 최대값과 최대값의 인덱스 ; x.max(dim=0)
values, indices = torch.min(x, dim=0) # 최소값과 최소값의 인덱스 ; x.min(dim=0)

abs_x = torch.abs(x) 

z = torch.argmax(x, dim=0) # 최대값의 위치(인덱스)
z = torch.argmin(x, dim=0) # 최소값의 위치(인덱스)


mean_x = torch.mean(x.float(), dim=0) # 텐서 요소들의 평균값 

z = torch.eq(x, y) # 텐서 요소별로 서로 같은지 비교 


sorted_y , indices = torch.sort(y, dim=0, descending=False)  # 0축 방향으로 오름 차순으로 정리 


z = torch.clamp(x, min=0, max=10)  # 0보다 작은 요소는 모두 0으로, 10보다 큰 요소는 모두 10으로 고정(clamp)


x = torch.tensor( [1, 0, 1, 0, 0 ], dtype=torch.bool)

z = torch.any(x)  # 텐서 요소들 중 하나라도 True 이면 True를 반환               
z = torch.all(x)  # 텐서의 모든 요소들이 True 여야만 True 반환 



# ============================================================= #
#                        Tensor Indexing                        #
# ============================================================= #
# %% 03. 텐서 인덱싱 
batch_size = 10 
