#-*- coding: utf-8 -*-

"""
(ref) (ref) https://youtu.be/rAdLwKJBvPM?t=1316
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/albumentations_tutorial/segmentation.py
(ref) https://github.com/DoranLyong/ResNet-tutorial/blob/main/ResNet_pytorch/ResNet18_for_CIFAR-10.py

* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""

"""
Segmentation task 를 위한 Image augmentation 튜토리얼 
    * RGB 이미지도 변환하면서 segmentation mask 도 동일하게 변환시키기 


1. Load Image

2. Define the transformation options

3. Apply the transformations

4. Visualization 
"""


#%% 임포트 패키지  
import os.path as osp
import os

import cv2
import numpy as np
from PIL import Image 
import albumentations as A    # Albumentations 라이브러리 임포트 

from utils import plot_examples, visualize   # 이미지 데이터를 표현하기 위해 정의한 함수 





# ================================================================= #
#                          1. Load Image                            #
# ================================================================= #
# %% 01. 이미지 로드 

img = Image.open("images/elon.jpeg")
mask = Image.open("images/mask.jpeg")
mask2 = Image.open("images/second_mask.jpeg")


visualize(img)
visualize(mask)
visualize(mask2)



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
    ]
)




# ================================================================= #
#                    3. Apply the transformations                   #
# ================================================================= #
# %% 03. 이미지에 변환 옵션 적용하기 


img_list = [img]    # 변환된 이미지를 저장할 리스트 

img = np.array(img)     # 이미지 데이터를 <PIL> 객체에서 <Numpy> 객체로 변환 
                        # 변환 연산을 적용하기 위해서는 <Numpy> 객체여야 함 
mask = np.array(mask)   # np.asarray(mask), np.array(mask)
mask2 = np.array(mask2)                           


for i in range(5):   # 총 4번 순회하면서 변환 적용하기 

    augmentations = transform(image= img, masks=[mask, mask2])  # 변환 적용 
    
    augmented_img = augmentations["image"]   # 변환된 이미지 가져오기 
    augmented_masks = augmentations["masks"]


    img_list.append(augmented_img)   
    img_list.append(augmented_masks[0])   # for the 'mask'
    img_list.append(augmented_masks[1])   # for the 'mask2'



# ================================================================= #
#                        4. Visualization                           #
# ================================================================= #
# %% 04. 변환된 이미지들 표출 

plot_examples(img_list, bboxes=None)   # 변환된 이미지 전부 표출됨 


# %%
