# -*- coding: utf-8 -*-
"""
(ref) https://youtu.be/WAciKiDP2bo
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial4-convnet.py


* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'
* (ref) https://blog.naver.com/cheeryun/221685740941
"""



"""
Sequential, Functional, Subclassing API 방식에 대해 알고 싶으면 아래 링크 참고: 

    * Creating-Model methods ; (ref) https://blog.naver.com/cheeryun/221912941277


CNN 네트워크를 구성하고 ,
cifar10 데이터셋으로 학습하기:

1. Set device 
2. Create CNN model   
    - Sequential API 방식으로 만들기 
    - Functional APU 방식으로 만들기 

3. Load Data (cifar10)
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
from tensorflow.keras.datasets import cifar10


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


layer_list = [  keras.Input(shape=(32, 32, 3)),
                layers.Conv2D(32, 3, padding="valid", activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(128, 3, activation="relu"),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(10),
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
def my_model():
    """ 레이어 설계 
    """
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    
    outputs = layers.Dense(10)(x)

    """ 모델 초기화 
    """
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model



model = my_model()

print(model.summary())  # 모델 구조 출력 



# ================================================================= #
#                      3.  Load Data (cifar-10)                        #
# ================================================================= #
# %% 03. cifar-10 데이터 로드 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0



# ================================================================= #
#                       4. Model Compile                            #
# ================================================================= #
# %% 04. 모델 컴파일 

model.compile(  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                optimizer=keras.optimizers.Adam(lr=3e-4), 
                metrics=["accuracy"],
            )



# ================================================================= #
#                            5. Training                            #
# ================================================================= #
# %% 05. 학습 실행 

model.fit(x_train, y_train, batch_size=64, epochs=150, verbose=2)


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
