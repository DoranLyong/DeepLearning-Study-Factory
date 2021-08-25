# 05.2 – [But what are these EBMs used for?](https://youtu.be/eJeJWWEo7cE) 

(Q) What do we use these EBMs for?  (어따써!!!) 

아래에서 각 분야별 활용 예시 설명.

* one-to-one 함수 구조에서 
* one-to-many 함수 구조에서 

### 1. [Energy interpretation of classifiers](https://youtu.be/eJeJWWEo7cE?t=1) 



### 2. [Self-supervised learning](https://youtu.be/eJeJWWEo7cE?t=130) 



### 3. [Video compression as latent collection](https://youtu.be/eJeJWWEo7cE?t=300)



### 4. [Interpolation and extrapolation](https://youtu.be/eJeJWWEo7cE?t=492) 

- [extrapolation(보외법)](https://www.google.com/search?q=extrapolation&oq=extrapolation&aqs=chrome..69i57j0i512l9.229j0j1&sourceid=chrome&ie=UTF-8) - future prediction 같은 것 
-  In the high dimensions, all of machine learning is ```extrapolation```!
  - 저 차원에서는 linear regression 처럼 interpolation으로 작용하지 
  - 하지만, 만약 고차원에서는 대부분 extrapolation 이다 
    - Imagine you are in a space of images. You have a color image (256 x 256) which has 65,536 pixels. 
    - So, you are in 65,536-dimensional input space 
    - Even if you have a million samples, you're only covering a tiny portion of the dimensions of that space (즉, 65,536 이란 방대한 공간에서 실제 이미지가 차지하는 부분은 매우 한정적이다)
    - Those images are in a tiny sliver of surface among the space of all possible combinations of values of pixels 
    - So, what you're doing when you show the system a new image. It's very unlikely that this image is a linear combination of previous images, which means what you're doing is extrapolation not interpolation 







***

### Reference 

[1] [NYU-DLSP20, website](https://atcold.github.io/pytorch-Deep-Learning/) / 

[2] [Energy-Based Models](https://atcold.github.io/pytorch-Deep-Learning/en/week07/07-1/) /  



















