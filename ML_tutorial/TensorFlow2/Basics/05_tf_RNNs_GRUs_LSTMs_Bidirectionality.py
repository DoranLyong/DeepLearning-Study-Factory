# -*- coding: utf-8 -*-
"""
(ref) https://youtu.be/Ogvd787uJO8
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial6-rnn-gru-lstm.py


* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'
* (ref) https://blog.naver.com/cheeryun/221685740941
"""



"""
Sequential, Functional, Subclassing API 방식에 대해 알고 싶으면 아래 링크 참고: 

    * Creating-Model methods ; (ref) https://blog.naver.com/cheeryun/221912941277


RNNs, GRUs, LSTMs, Bidirectional-LSTM 네트워크를 구성하고 ,
MNIST 데이터셋으로 학습하기:

1. Set device 
2. Create model   
    - Simple RNN  
    - GRU  
    - Bidirectional LSTM   


3. Load Data (MNIST)
4. Model Compile
5. Train network
6. Check accuracy on test to see how good our model
7. prediction (=inference)    
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
#                          2-1. Simple RNN                          #
# ================================================================= #
# %% 02-1. 심플한 RNN 생성하기 using Sequential API (A bit more flexible)

RNN_list = [    keras.Input(shape=(None, 28)),
                layers.SimpleRNN(512, return_sequences=True, activation="relu"),
                layers.SimpleRNN(512, activation="relu"),
                layers.Dense(10)
            ]


model = keras.Sequential(RNN_list)

print(model.summary())  # 모델 구조 출력 

#keras.utils.plot_model(model, to_file="first_Sequential_model.png", show_shapes=True)   # 모델 그래프 출력; 
                                                                                        # (ref) https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
                                                                                        # (ref) https://gldmg.tistory.com/71



# ================================================================= #
#                          2-2. Simple RNN                          #
# ================================================================= #
# %% 02-2. 심플한 RNN 생성하기 using Sequential API (A bit more flexible)

RNN_list = [    keras.Input(shape=(None, 28)),
                layers.SimpleRNN(256, return_sequences=True, activation="tanh"),
                layers.SimpleRNN(256),
                layers.Dense(10)
            ]


model = keras.Sequential(RNN_list)

print(model.summary())  # 모델 구조 출력 




# ================================================================= #
#                            2-3. GRU                               #
# ================================================================= #
# %% 02-3. GRU 생성하기 using Sequential API (A bit more flexible)

GRU_list = [    keras.Input(shape=(None, 28)),
                layers.GRU(256, return_sequences=True, activation="relu"),
                layers.GRU(256),
                layers.Dense(10)
            ]


model = keras.Sequential(GRU_list)

print(model.summary())  # 모델 구조 출력 


# ================================================================= #
#                    2-4. Bidirectional LSTM                        #
# ================================================================= #
# %% 02-4. Bidirectional LSTM 생성하기 using Sequential API (A bit more flexible)

BiLSTM_list = [ keras.Input(shape=(None, 28)),
                layers.Bidirectional(layers.LSTM(256, return_sequences=True, activation="relu")),
                layers.LSTM(256, name="lstm_layer2"),
                layers.Dense(10)
              ]


model = keras.Sequential(BiLSTM_list)

print(model.summary())  # 모델 구조 출력 



# ================================================================= #
#                    2-5. Bidirectional LSTM                        #
# ================================================================= #
# %% 02-5. Bidirectional LSTM 생성하기 using Sequential API (A bit more flexible)

BiLSTM_list = [ keras.Input(shape=(None, 28)),
                layers.Bidirectional(layers.LSTM(256, return_sequences=True, activation="relu")),
                layers.Bidirectional(layers.LSTM(256, name="lstm_layer2")),
                layers.Dense(10)
              ]


model = keras.Sequential(BiLSTM_list)

print(model.summary())  # 모델 구조 출력 




# ================================================================= #
#                      3.  Load Data (MNIST)                        #
# ================================================================= #
# %% 03. MNIST 데이터 로드 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
# x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
x_train = x_train.reshape([-1, 28, 28]).astype("float32") / 255.0
x_test = x_test.reshape([-1, 28, 28]).astype("float32") / 255.0



# ================================================================= #
#                       4. Model Compile                            #
# ================================================================= #
# %% 04. 모델 컴파일 

model.compile(  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=keras.optimizers.Adam(lr=1e-3), 
                metrics=["accuracy"],
            )



# ================================================================= #
#                            5. Training                            #
# ================================================================= #
# %% 05. 학습 실행 

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)


# ================================================================= #
#                             6. Test                               #
# ================================================================= #
# %% 06. 모델 평가 

model.evaluate(x_test, y_test, batch_size=64, verbose=2)




# ================================================================= #
#                    7. Prediction (=inference)                     #
# ================================================================= #
# %% 07. 학습된 모델을 통한 예측

query = x_test[:1]

y_prediction = model.predict(query)   # (ref) https://blog.naver.com/cheeryun/221927717487

print(query.shape)
print(y_prediction)

print(f"argmax: {y_prediction.argmax()}")  # (ref) https://rfriend.tistory.com/356
print(f"GT_label: {y_test[0]}")
