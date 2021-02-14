#-*- coding: utf-8 -*-

"""
(ref) https://youtu.be/rAdLwKJBvPM?t=1316
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/albumentations_tutorial/detection.py
(ref) https://github.com/DoranLyong/ResNet-tutorial/blob/main/ResNet_pytorch/ResNet18_for_CIFAR-10.py

* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""

"""
Detection task 를 위한 Image augmentation 튜토리얼 
    * RGB 이미지도 변환하면서 bbox 좌표도 동일하게 변환시키기 


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

img = cv2.imread("images/cat.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # <PIL> 클래스는 RGB,  <cv2> 클래스는 BGR 순서니까 


bboxes = [
    [13, 170, 224, 410], # Pascal_voc 형태의 bbox 좌표가 주어진다면 
                        """
                        지금은 이미지 하나에 object instance 가 하나만 있어서 bbox 좌표 리스트가 하나 뿐이지만,
                        여러 object가 있다면 bbox 좌표 리스트도 여러개가 될 것. 

                        이때를 표현하기 위해 자료구조를 list-in-list 형태로 만듬. 
                        """
    ]    


"""
Pascal_voc := (x_min, y_min, x_max, y_max)

YOLO := (c_x, c_y, width, heigh)

COCO :=  ~~~ 

"""



visualize(img)


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
    ], 
    bbox_params=A.BboxParams(   format="pascal_voc",  # bbox 좌표는 Pascal_voc 포멧을 따름 
                                min_area=2048,        # 변환하다 객체가 짤려서 안 보일 수 있다. 따라서, 적어도 bbox 영역이 2048 pixels 이상인 것만 가져와라 
                                min_visibility=0.3,   # 전체 이미지에서 bbox 는 30%는 보일 것 
                                label_fields=[]
                            )
)




# ================================================================= #
#                    3. Apply the transformations                   #
# ================================================================= #
# %% 03. 이미지에 변환 옵션 적용하기 


img_list = [img]    # 변환된 이미지를 저장할 리스트 

img = np.array(img)     # 이미지 데이터를 <PIL> 객체에서 <Numpy> 객체로 변환 
                        # 변환 연산을 적용하기 위해서는 <Numpy> 객체여야 함 


saved_bboxes = [bboxes[0]]  # 이미지상의 0번째 객체의 bbox 좌표                  


for i in range(8):   # 총 4번 순회하면서 변환 적용하기 

    augmentations = transform(image= img, bboxes=bboxes)  # 변환 적용 
    
    augmented_img = augmentations["image"]   # 변환된 이미지 가져오기 
    
    if len(augmentations["bboxes"]) == 0:
        """
        이미지상에 bbox를 사진 객체가 없다면, 
        다음으로 념김. 
        """
        continue


    img_list.append(augmented_img)   
    saved_bboxes.append(augmentations["bboxes"][0])   # 변환된 bbox 좌표 가져오기 



# ================================================================= #
#                        4. Visualization                           #
# ================================================================= #
# %% 04. 변환된 이미지들 표출 

plot_examples(img_list, bboxes=saved_bboxes)   # 변환된 이미지 전부 표출됨 


# %%
print(saved_bboxes[1])  # 변환된 bbox 좌표 출력하기 
# %%
