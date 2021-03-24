# -*- coding: utf-8 -*-
"""
(ref) https://youtu.be/HPjBY1H-U4U 
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial2-tensorbasics.py
"""
#%%
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # 에러 로그 삭제 ; (ref) https://yongyong-e.tistory.com/62
                                            #                 (ref) https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
#%% 임포트 토치 
import tensorflow as tf



# ================================================================= #
#                        Initializing Tensor                        #
# ================================================================= #
# %% 01. 텐서 생성하기 
x = tf.constant(4, shape=(1, 1), dtype=tf.float32)     # scalar = 1D-tensor 
x = tf.constant([[1, 2, 3], [4, 5, 6]], shape=(2, 3))  # matrix = 2D-tensor  



# %% 텐서를 초기화하는 다른 방법들 
x = tf.eye(3) # 3x3 identity tensor (I)
x = tf.ones((4, 3)) # 4x3 ones 
x = tf.zeros((3, 2, 5)) 
x = tf.random.uniform((2, 2), minval=0, maxval=1)  #  2x2,  0 부터 1사이에서 uniform sampling
x = tf.random.normal((3, 3), mean=0, stddev=1)  # 3x3, 평균=0, std=1 인 정규분포로 샘플링 
print(tf.cast(x, dtype=tf.float64)) # tf.cast() 자료형 새로 씌우기; (ref) https://www.tensorflow.org/api_docs/python/tf/cast?hl=ko
                                    
x = tf.range(9) # int32 자료형으로 [0, 9) 범위에서 1스텝으로(=9 개) 시퀀스 텐서를 만든다 
x = tf.range(start=0, limit=10, delta=2) # [0, 10) 범위에서 2스텝으로 시퀀스 텐서를 만든다 




# =============================================================================== #
#                        Tensor Math & Comparison Operations                      #
# =============================================================================== #
# %% 02. 텐서 연산 (Math & Comparison Operations)
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])


#%% 덧셈(addition)
z = tf.add(x, y)
z = x + y


# %% 뺄셈(subtraction)
z = tf.subtract(x, y)
z = x - y 


# %% 나눗셈(division)
z = tf.divide(x, y)
z = x / y


# %% 곱셈(multiplication)
z = tf.multiply(x, y)
z = x * y


# %% 벡터 내적 (dot-product)
z = tf.tensordot(x, y, axes=1)  # (ref) https://www.tensorflow.org/api_docs/python/tf/tensordot



# %% 벡터 외적 (cross-product)
z = tf.tensordot(x, y, axes=0)  # (ref) https://www.tensorflow.org/api_docs/python/tf/tensordot


# %%
x = tf.random.normal((2, 3))
y = tf.random.normal((3, 2))


# %% 행렬 곱셈(matrix multiplication)
z = tf.matmul(x, y)   # (ref) https://chan-lab.tistory.com/tag/tf.multiply%20vs%20tf.matmul
z = x @ y   # (ref) https://www.tensorflow.org/api_docs/python/tf/linalg/matmul




# ============================================================= #
#                        Tensor Indexing                        #
# ============================================================= #
# %%
x = tf.constant([0, 1, 2, 3, 4, 5, 6, 7])


# %% 03. 텐서 인덱싱 
print(x[:])
print(x[1:])
print(x[1:3])
print(x[::2])   # 처음부터 끝까지 2스텝으로 
print(x[::-1])  # 처음부터 끝까지 역순으로 


# %% 좀더 기교있는 인덱싱 (Facny indexing)
indices = tf.constant([0, 3])
x_indices = tf.gather(x, indices)  # indices 0, 3의 요소 가져오기 


# %%
x = tf.constant([[1, 2], [3, 4], [5, 6]])

print(x)
print(x[0, :])
print(x[0:2, :])


# ============================================================= #
#                        Tensor Reshaping                       #
# ============================================================= #
# %% 04. 텐서 형태 바꾸기 
x = tf.range(9)   # 요소 9개 초기화 


#%%
x = tf.reshape(x, (3, 3))  # 요소 9개를 3x3 형태로 변환 
print(x)

#%%
x = tf.transpose(x, perm=[1, 0])  # permute the dimensions according to [1, 0]  (ref) https://www.tensorflow.org/api_docs/python/tf/transpose
                                  # 축(axes)을 1, 0 순으로 교환 

print(x)

# %%
