# -*- coding: utf-8 -*-
"""
(ref) https://youtu.be/gRRGr_tJnAA
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial7-indepth-functional.py


* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'
* (ref) https://blog.naver.com/cheeryun/221685740941
"""



"""
Sequential, Functional, Subclassing API 방식에 대해 알고 싶으면 아래 링크 참고: 

    * Creating-Model methods ; (ref) https://blog.naver.com/cheeryun/221912941277


다중 출력하는 CNN 모델을 생성하고,
MNIST 데이터셋으로 학습하기:

1. Set device 
2. Hyperparameters
3. Create model   
    - Functional API 방식 더 알아보기 (다중 입출력 모델)

4. DownLoad & UpLoad Data (MNIST) 
    -(ref) https://www.kaggle.com/dataset/eb9594e5b728b2eb74ff8d5e57a9b74634330bfa79d9195d6ebdc7745b9802c3?select=train_images

5. Model Compile
6. Train network
7. Check accuracy on test to see how good our model
8. prediction (=inference)    
"""



#%%
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # 에러 로그 삭제 ; (ref) https://yongyong-e.tistory.com/62
                                            #                 (ref) https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
import os.path as osp 

import pandas as pd     # Use Pandas to load dataset from csv file                                             

#%% 임포트 tf 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
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
#                       2. Hyperparameters                          #
# ================================================================= #
# %% 02. 하이퍼파라미터 설정 

BATCH_SIZE = 64
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-3



# ================================================================= #
#                      3. CNN for multiple outputs                  #
# ================================================================= #
# %% 03. 다중 출력하는 CNN 생성하기 using Functional API (A bit more flexible)
""" (ref) https://blog.naver.com/cheeryun/221912941277
"""


def my_model():
    """ 레이어 설계 
    """
    inputs = keras.Input(shape=(64, 64, 1))
    x = layers.Conv2D(  filters=32,
                        kernel_size=3,
                        padding="same",
                        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                    )(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.Conv2D(64, 3, kernel_regularizer=regularizers.l2(WEIGHT_DECAY),)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D( 64, 3, activation="relu", kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                    )(x)

    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)

    output1 = layers.Dense(10, activation="softmax", name="first_num")(x)
    output2 = layers.Dense(10, activation="softmax", name="second_num")(x)

    """ 모델 초기화 
    """
    model = keras.Model(inputs=inputs, outputs=[output1, output2])   # 다중 출력하는 CNN
    return model



model = my_model()

print(model.summary())  # 모델 구조 출력 



#keras.utils.plot_model(model, to_file="first_Sequential_model.png", show_shapes=True)   # 모델 그래프 출력; 
                                                                                        # (ref) https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
                                                                                        # (ref) https://gldmg.tistory.com/71





# ================================================================= #
#                      3.  Load Data (MNIST)                        #
# ================================================================= #
# %% 03. MNIST 데이터 로드 

dataset_path = osp.join(os.getcwd(), 'dataset', 'multi-digit-mnist' )

train_df = pd.read_csv(osp.join(dataset_path, "train.csv"))
test_df = pd.read_csv(osp.join(dataset_path, "test.csv"))

train_images = [osp.join(dataset_path, 'train_images', img ) for img in train_df.iloc[:, 0].values]
test_images  = [osp.join(dataset_path, 'train_images', img ) for img in test_df.iloc[:, 0].values]


train_labels = train_df.iloc[:, 1:].values
test_labels = test_df.iloc[:, 1:].values



#%%
def read_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)

    """In older versions you need to set shape in order to avoid error
        on newer (2.3.0+) the following 3 lines can safely be removed
    """
    image.set_shape((64, 64, 1))
    label[0].set_shape([])
    label[1].set_shape([])

    labels = {"first_num": label[0], "second_num": label[1]}
    return image, labels

#%%
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = (   train_dataset.shuffle(buffer_size=len(train_labels))
                    .map(read_image)
                    .batch(batch_size=BATCH_SIZE)
                    .prefetch(buffer_size=AUTOTUNE)
                )

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = (    test_dataset.map(read_image)
                    .batch(batch_size=BATCH_SIZE)
                    .prefetch(buffer_size=AUTOTUNE)
                )




# ================================================================= #
#                       4. Model Compile                            #
# ================================================================= #
# %% 04. 모델 컴파일 

model.compile(  loss=keras.losses.SparseCategoricalCrossentropy(),
                optimizer=keras.optimizers.Adam(LEARNING_RATE), 
                metrics=["accuracy"],
            )



# ================================================================= #
#                            5. Training                            #
# ================================================================= #
# %% 05. 학습 실행 

model.fit(train_dataset, epochs=5, verbose=2)


# ================================================================= #
#                             6. Test                               #
# ================================================================= #
# %% 06. 모델 평가 

model.evaluate(test_dataset, verbose=2)




# ================================================================= #
#                    7. Prediction (=inference)                     #
# ================================================================= #
# %% 07. 학습된 모델을 통한 예측

#query = x_test[:1]
#
#y_prediction = model.predict(query)   # (ref) https://blog.naver.com/cheeryun/221927717487
#
#print(query.shape)
#print(y_prediction)
#
#print(f"argmax: {y_prediction.argmax()}")  # (ref) https://rfriend.tistory.com/356
#print(f"GT_label: {y_test[0]}")
