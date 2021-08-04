# 02 – [Neural nets: rotation and squashing](https://youtu.be/0TdAmZUMj2k)


### (Q) [In a very high dimensional space, there are less local minimum or less likely to have local minimum. Can you elaborate?](https://youtu.be/0TdAmZUMj2k)
* the weight matrix of the output method 



### 1. [Affine transformation and non-linearities](https://youtu.be/0TdAmZUMj2k?t=151)
* linear transformation 
* non-linear transformation 


### 2. [Affine transformation: intuition](https://youtu.be/0TdAmZUMj2k?t=222)
* 2D 좌표상에 벡터는 어떻게 표현할까? → 점으로 표현하기 
* 2D 벡터에 linear transformation을 적용하면 ```원점을 기준으로``` 변한다 
  *  transformation matrix 가 negative ⇒  reflection 
  *  기타 변환 = {scaling, rotation, shearing}
* linear transformation을 확장하면 affine transformation 이다. 무엇이 추가 되었지? 
  * shifting (= translation)
* [고양이랑 개 그림이 있다.](https://youtu.be/0TdAmZUMj2k?t=414) 이 두 카테고리의 데이터가 huge dimension에서 어떻게 분포해 있을가?
  * 거의 비슷하게 생긴 이미지라 통계적으로(statistically) 두 카테고리는 거진 뭉쳐진 상태로 존재한다 
  * 마치 spiral data distribution 같이 뒤 섞여 있음

* 비선형 변환 ⇒ [squashing](https://www.google.com/search?q=squashing&bih=2075&biw=1342&hl=ko&sxsrf=ALeKk000YEYEotrePamX0EIBhcebEO6MKw%3A1625178039505&source=hp&ei=tz_eYJu_HPK1mgftkZjQDw&iflsig=AINFCbYAAAAAYN5Nx2OtCizmlyKtpoX-ir7h8pz6-loy&oq=squashing&gs_lcp=Cgdnd3Mtd2l6EAMyBAgjECcyBQgAEMsBMgUIABDLATIFCAAQywEyBQgAEMsBMgUIABDLATIFCAAQywEyBQgAEMsBMgUIABDLATIFCAAQywFQhAhYhAhg2gpoAHAAeACAAWSIAWSSAQMwLjGYAQCgAQKgAQGqAQdnd3Mtd2l6&sclient=gws-wiz&ved=0ahUKEwjb9vT988LxAhXymuYKHe0IBvoQ4dUDCAc&uact=5)
* [What the neural network does are basically two things:](https://youtu.be/0TdAmZUMj2k?t=684)

  * ```rotating```(affine transformation) & ```squashing``` data(non-linear)
  * 이교수 관점에서 rotating은  affine transformation을 말한다 
  * 그냥 '이리 저리 단순히 돌리고' 라는 표현인가 보다?
  

※ ```squashing(= twisting)``` : changing things in a non-linear way

### 3. [Summary slide](https://youtu.be/0TdAmZUMj2k?t=827)

* Neural nets 은 결국 벡터를 rotation → squashing →rotation → squashing →...  을 반복해서 질 좋은 feature vector를 찾는 것 

### 4. [Jupyter and PyTorch](https://youtu.be/0TdAmZUMj2k?t=846)


### 5. [Input data](https://youtu.be/0TdAmZUMj2k?t=1127)


### 6. [Coding a 2×2 linear transformation ](https://youtu.be/0TdAmZUMj2k?t=1479)& Gilbert Strang


### 7. [Coding a 2×2 linear transformation]() w/ PyTorch

### 8. [The Neural Network, A Visual Introduction](https://youtu.be/UOvPeC8WOt8); 코드 따라 해보기 

***

### Reference 

[1] [The True Power of the Matrix (Transformations in Graphics) - Computerphile](https://youtu.be/vQ60rFwh2ig) / [three basic linear transformation](https://youtu.be/vQ60rFwh2ig?t=93) = { scaling, translation, rotation } & [필기](https://github.com/DoranLyong/DeepLearning-Study-Factory/blob/master/Study_resource/NYU_DL/02_Neural_nets-rotation_and_squashing/The%20true%20power%20of%20the%20matrix.pdf)<br/>

[2] [The Neural Network, A Visual Introduction](https://youtu.be/UOvPeC8WOt8) / <br/>

[3] [Essence of linear algebra](https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) / <br/>

[4] [행렬과 선형변환, 공돌이의 수학정리노트](https://angeloyeo.github.io/2019/07/15/Matrix_as_Linear_Transformation.html) / Linear layer에서 어떤 변환이 일어나는 지(w/o activation function) 이미지화 하기<br/>

