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




# %%


# =============================================================================== #
#                        Tensor Math & Comparison Operations                      #
# =============================================================================== #
# %%
