#-*- coding: utf-8 -*-

"""
(ref) https://youtu.be/rAdLwKJBvPM?t=1316
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/albumentations_tutorial/full_pytorch_example.py
(ref) https://github.com/DoranLyong/ResNet-tutorial/blob/main/ResNet_pytorch/ResNet18_for_CIFAR-10.py
(ref) https://github.com/DoranLyong/DeepLearning_model_factory/blob/master/ML_tutorial/PyTorch/Basics/07_custom_dataset_image.py

* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""

"""
Albumentations 라이브러리를 PyTorch Tensor 에 사용할 수 있게 만들기 


1. Create custom dataset object for torch.Tensor 

2. Define the transformation options

3. Load your custom dataset

4. Visualization 
"""


#%% 임포트 패키지  
import os.path as osp
import os

import cv2
import numpy as np
from PIL import Image 
import albumentations as A    # Albumentations 라이브러리 임포트 
from albumentations.pytorch import ToTensorV2   # Convert image and mask to torch.Tensor # (ref) https://albumentations.ai/docs/api_reference/pytorch/transforms/#albumentations.pytorch.transforms.ToTensor
import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
from torch.utils.data import Dataset    # 가져다쓸 데이터셋 객체를 지칭하는 클래스 (ref) https://huffon.github.io/2020/05/26/torch-data/

from utils import plot_examples, visualize   # 이미지 데이터를 표현하기 위해 정의한 함수 





# ================================================================= #
#                    1. Create custom dataset object                #
# ================================================================= #
# %% 01. 데이터셋 객체 생성하기 
"""
(dataset) https://www.kaggle.com/dataset/c75fbba288ac0418f7786b16e713d2364a1a27936e63f4ec47502d73d6ef30ab 
에서 데이터을 먼저 받는다. 

데이터셋은 ./dataset/archive 에 위치시킨다. 


# Dataset 클래스와 DataLoader 클래스의 관계:
- DataLoader와 Dataset 클래스의 상속관계 (ref) https://hulk89.github.io/pytorch/2019/09/30/pytorch_dataset/
- Dataset 클래스로 커스텀 데이터 셋을 만든다 => 생성된 데이터셋을 DataLoader 클래스로 전달해서 불러온다 (ref) https://wikidocs.net/57165
- (ref) https://doranlyong-ai.tistory.com/42
"""

class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        super(ImageFolder, self).__init__()
        """
        가져다쓸 데이터셋의 정보를 초기화한다. 
        """

        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(osp.join(root_dir, name))
            self.data += list(zip(files, [index]*len(files)))

    def __len__(self):
        """
        초기화된 객체가 컨테이너 자료형을 가지고 있으면, 그것의 길이를 반환한다

        __len__() 매직 함수를 사용하면 내장 함수 len()을 사용할 수 있다 

        (ref) https://dgkim5360.tistory.com/entry/python-duck-typing-and-protocols-why-is-len-built-in-function
        (ref) https://kwonkyo.tistory.com/234
        (ref) https://medium.com/humanscape-tech/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9D%98-%EC%8A%A4%ED%8E%98%EC%85%9C-%EB%A9%94%EC%84%9C%EB%93%9C-special-method-2aea6bc4f2b9

        """
        return len(self.data) # 로드된 데이터의 개수(길이) 를 반환한다 


    def __getitem__(self, index):
        """
        데이터셋 시퀀스에서 특정 index에 해당하는 아이템을 가져온다 (= 객체에 indexing 기능을 사용할 수 있음). 

        (ref) http://hyeonjae-blog.logdown.com/posts/776615-python-getitem-len
        """

        img_file, label = self.data[index]
        root_and_dir = osp.join(self.root_dir, self.class_names[label])
        image = np.array(Image.open(osp.join(root_and_dir, img_file)))

        if self.transform is not None:
            augmentations = self.transform(image=image)   # 변환 적용 (앞서 연습한 코드 확인)
            image = augmentations["image"]  # 변환된 이미지 가져오기 

        return image, label





# ================================================================= #
#             2. Define the transformation options                  #
# ================================================================= #
# %% 02. 변환 옵션 정의하기 
"""
torchvision 의 transfomr가 사용방법이 비슷함 
"""
transform = A.Compose(   # 아래 옵션들을 함께 적용하기 
    [
        A.Resize(width=1920, height=1080),        # 이미지 크기를 (H, W)=(1080, 1920) 으로 변환 
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),  # 이미지를 회전시킬 때 경계 프레임 부분은 움직이지 말구
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),

        A.OneOf(    # 이중에 랜덤으로 하나의 옵션을 골라서 적용하기 
        [  
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
        ], p=1.0),  # 100% 확률로 이중에 하나는 골라야함 

        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),

        ToTensorV2(),  # Albumentations to torch.Tensor

    ], 
)




# ================================================================= #
#                    3.  Load your custom dataset                   #
# ================================================================= #
# %% 03. 커스텀 데이터 로드 

"""
랜덤 발생 기준을 시드로 고정함. 
그러면 shuffle=True 이어도, 언제나 동일한 방식으로 섞여서 동일한 데이터셋을 얻을 수 있음. 
"""
torch.manual_seed(42)


dataset = ImageFolder(  root_dir="cat_dogs", 
                        transform = transform,  # Albumentations 라이브러리를 활용한 변환 적용 
                        
                    )





# ================================================================= #
#                        4. Visualization                           #
# ================================================================= #
# %% 04. 변환 확인 

for x,y in dataset:
    print(f"{x.shape}, {y}")   # (C, H, W) 순으로 torch.Tensor 형태로 바뀐걸 확인 
                                # 고양이 := 1,  고양이가 아니면 := 0
# %%
