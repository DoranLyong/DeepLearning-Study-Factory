# coding=<utf-8> 
"""
(ref) https://youtu.be/x9JiIFvlUwk
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorbasics.py
(ref) https://jovian.ai/aakashns/01-pytorch-basics

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
features = 25 
x = torch.rand( (batch_size, features) )



# %% Get the first examples features 
print(x[0].shape)  # this is the same as doing x[0, :]

# %% Get the first feature for all examples 
print(x[:, 0].shape)  # shape[10]


# %% For example: Want to access third example in the batch and the first ten features 
print( x[2, 0:10].shape ) # shape[10] ; 0:10 -> [0, 1, 2, ..., 9]



# %% 특정 위치에 할당하기(assign certain elements)
x[0, 0] = 100 


# %% 좀더 기교있는 인덱싱 (Facny indexing)
x = torch.arange(10)
indices = [2, 5, 8]
print( x[indices] )   # x[indices] = [2, 5, 8]


x = torch.rand( (3,5) )
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])  # Gets second row fourth column (1,4) and first row first column (0, 0) 


# %% 좀더 고급진 인덱싱 (More advanced indexing)
x = torch.arange(10) #

print( x[ (x<2) | (x>8)] )  # 조건부 인덱싱; will be [0, 1, 9] ; 2보다 작고 8보다 큰 요소들 인덱싱 
print( x[x.remainder(2) == 0] ) # will be [0, 2, 4, 6, 8] ; 2로 나누었을 때 나머지가 0인 요소들 인덱싱 




# %% 기타 유용한 연산 (useful operations)
print( 
        torch.where( x>5, x, x*2)    # all values x > 5 yield x, else x*2
)                                    # 5보다 크면 x, 작으면 x*2 로 바꿔라 


x = torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()  # x = [0, 1, 2, 3, 4] 


print(
    x.ndimension()  # The number of dimensions, in this case 1. if x.shape is 5x5x5 ndim would be 3
)                   # 텐서 x의 차원축 개수 반환 


x = torch.arange(10)
print(
    x.numel()   # The number of elements in x (in this case it's trivial because it's just a vector)
                # 텐서 x의 요소 개수 세기 (the number of elements = numel )
)  



# ============================================================= #
#                        Tensor Reshaping                       #
# ============================================================= #
# %% 04. 텐서 형태 바꾸기 
x = torch.arange(9) 

x_3x3 = x.view(3, 3) 
x_3x3 = x.reshape(3, 3)

"""
view() 와 reshape() 의 기능은 비슷하다. 
차이점은 view 는 기존 텐서에서 연속적인 메모리에 있는 요소들의 형태만 바꿔서 가져오는거고 
reshape 은 새로 복사해서 형태를 바꾸는 것? 

(ref; contigious vs non-contigious) https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
(ref) https://sanghyu.tistory.com/3
(ref) https://subinium.github.io/pytorch-Tensor-Variable/
"""


# %% Contiguous examples 
"""
즉, 텐서 x 의 요소들이 contiguous 하지 않다면(=연속적인 메모리에 없다면), view() 는 에러가 발생한다. 
"""
x = torch.arange(9)   # 요소들이 연속적인 메모리에 위치한다 

x_3x3 = x.view(3,3)   # 텐서 x의 요소들은 contiguous 하니까 view() 사용할 수 있다. 

y = x_3x3.t() # transpose 하면서 contiguous 했던 요소들의 위치가 뒤바뀐다; non-contiguous. (왜냐하면 배열도 본질적으로는 연속된 메모리로 표현하니까)

# print( y.view(9) )  # Error; 결국 y는 non-contiguous 하기 때문에 view() 를 사용할 수 없다. 

print( y.contiguous().view(9) ) # 텐서 y를 contiguous 하게 바꿔줘야만 view()를 사용할 수 있다. 



# %% 이어 붙이기 (concatenate)
x1 = torch.rand( (2, 5) )
x2 = torch.rand( (2, 5) )

print( torch.cat((x1, x2), dim=0).shape)  # Shape: 4x5  (dim0 축 방향으로 이어붙인다)
print( torch.cat((x1, x2), dim=1).shape)  # Shape: 2x10 (dim1 축 방향으로 이어붙인다)


# %% 텐서 펼치기 (unroll) - 텐서를 벡터로 만든다 
z = x1.view(-1)  # -1 will unroll everything 
print(z.shape)   # torch.Size([10]) 



# %% 특정 부분 빼고 전부 펼치기
batch = 64 
x = torch.rand( (batch, 2, 5) )  # 64 x 2 x 5 

z = x.view(batch, -1)   # 0번째 축은 batch 길이 만큼 두고 나머지는 펼친다(unroll)
print( z.shape )   # torch.Size([64, 10])


# %% 요수 축 위치 바꾸기 (permute)
z = x.permute(0, 2, 1) # 64 x 2 x 5  ->  64 x 5 x 2


# %% 텐서 자르기/분리하기; 덩어리 짓기 (chunk)
"""
텐서를 특정 축 방향으로 등분하기 (덩어리 짓기)
(ref) https://sanghyu.tistory.com/6
"""

x = torch.rand( (64, 6, 10) )  # 64x6x10

z = torch.chunk(x, chunks=2, dim=1)  # 텐서 x를 2 덩어리(chunk) 짓는데, dim1 축 방향으로 덩어리 지어라 

print(z[0].shape)  # 64x3x10
print(z[1].shape)  # 64x3x10


"""
두 개로 덩어리 지어졌다 (chunk=2)
"""


# %% 길이 1만큼 차원축 추가하기 (unsqueeze)
x = torch.arange(10)  # (10,)

print(x.unsqueeze(0).shape) # 0축 방향으로 길이 1만큼 차원을 추가한다; (10,) -> (1, 10) shape 
print(x.unsqueeze(1).shape) # 0축 방향으로 길이 1만큼 차원을 추가한다; (10,) -> (10, 1) shape 


# %% 반대로 길이 1인 차원축 짜부수기 (squeeze)
x = torch.arange(10).unsqueeze(0).unsqueeze(1)   # 1x1x10 shape 

z = x.squeeze(1)  # dim1 축 방향으로 길이가 1이면 짜부수기 -> 1x10 shape 