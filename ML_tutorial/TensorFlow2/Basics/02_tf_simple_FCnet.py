# -*- coding: utf-8 -*-
"""
(ref) https://youtu.be/pAhPiF3yiXI
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial3-neuralnetwork.py


* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'
* (ref) https://blog.naver.com/cheeryun/221685740941
"""



"""
Sequential, Functional, Subclassing API 방식에 대해 알고 싶으면 아래 링크 참고: 

    * Creating-Model methods ; (ref) https://blog.naver.com/cheeryun/221912941277


Fully-connected(FC) 네트워크를 구성하고,
MNIST 데이터셋으로 학습하기:

1. Set device 
2. Create a fully-connected network 
    - Sequential API 방식으로 만들기 
    - Functional APU 방식으로 만들기 

3. Load Data (MNIST)
4. Model Compile
5. Train network
6. Check accuracy on test to see how good our model
"""



#%%
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # 에러 로그 삭제 ; (ref) https://yongyong-e.tistory.com/62
                                            #                 (ref) https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
#%% 임포트 토치 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist



# ================================================================= #
#                         1. Set device                             #
# ================================================================= #
# %% 01. 프로세스 장비 설정 
physical_devices = tf.config.list_physical_devices('GPU')  # GPU 장치 목록 출력; (ref) https://stackoverflow.com/questions/58956619/tensorflow-2-0-list-physical-devices-doesnt-detect-my-gpu

if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

else: 
    print("No GPU")


# ================================================================= #
#    2-1. Create a fully-connected network  with Sequential API     #
# ================================================================= #
# %% 02-1. 심플한 FCnet 생성하기 using Sequential API (Very convenient, not very flexible)

#%% 방법 1 
model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)

print(model.summary())  # 모델 구조 출력 

#%% 방법 2
model = keras.Sequential()
model.add(keras.Input(shape=(28*28)))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(256, activation="relu", name="my_layer"))
model.add(layers.Dense(10))

print(model.summary())  # 모델 구조 출력 


# %% 방법 3 
layer_list = [  keras.Input(shape=(28*28)),
                layers.Dense(512, activation="relu"),
                layers.Dense(256, activation="relu", name="my_layer"),
                layers.Dense(10)
            ]

model = keras.Sequential(layer_list)

print(model.summary())  # 모델 구조 출력 

#keras.utils.plot_model(model, to_file="first_Sequential_model.png", show_shapes=True)   # 모델 그래프 출력; 
                                                                                        # (ref) https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
                                                                                        # (ref) https://gldmg.tistory.com/71

"""
방법 1, 2, 3 의 결과가 모두 동일 한 구조임을 확인. 
"""



# ================================================================= #
#    2-2. Create a fully-connected network  with Functional API     #
# ================================================================= #
# %% 02-2. 심플한 FCnet 생성하기 using Functional API (A bit more flexible)

""" 레이어 설계 
"""
inputs = keras.Input(shape=(28*28))
x = layers.Dense(512, activation="relu", name="first_layer")(inputs)
x = layers.Dense(256, activation="relu", name="second_layer")(x)
outputs = layers.Dense(10, activation="softmax")(x)

""" 모델 초기화 
"""
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())  # 모델 구조 출력 



# ================================================================= #
#                      3.  Load Data (MNIST)                        #
# ================================================================= #
# %% 03. MNIST 데이터 로드 

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0



# ================================================================= #
#                       4. Model Compile                            #
# ================================================================= #
# %% 04. 모델 컴파일 

model.compile(  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                optimizer=keras.optimizers.Adam(lr=1e-3), 
                metrics=["accuracy"],
            )



# ================================================================= #
#                            5. Training                            #
# ================================================================= #
# %% 05. 학습 실행 

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)


# ================================================================= #
#                             6. Test                               #
# ================================================================= #
# %% 06. 모델 평가 

model.evaluate(x_test, y_test, batch_size=32, verbose=2)