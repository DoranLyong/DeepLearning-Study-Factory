# 03 – [Tools, classification with neural nets, PyTorch implementation](https://youtu.be/EyKiYVwrdjE)


### 1. [Typora](https://youtu.be/EyKiYVwrdjE?t=46)
* an editor tool for ```Markdown``` & ```LATEX```

### 2. [Notion](https://youtu.be/EyKiYVwrdjE?t=95)

***

### 3. [Lecture begins](https://youtu.be/EyKiYVwrdjE?t=194)

* theme - ```how to training?``` 
  * [gradient descent](https://www.google.com/search?q=gradient+descent&sxsrf=ALeKk03j171yc33GKMSuP_iyObeq8B7BJw%3A1627313180103&ei=HNT-YMrsBaryhwOSpLHwCg&oq=gradi&gs_lcp=Cgdnd3Mtd2l6EAMYADIHCAAQsQMQQzIFCAAQsQMyAggAMgQIABBDMgQIABBDMgIIADICCAAyAggAMgIIADICCAA6BwgjELADECc6BwgAEEcQsAM6BAgjECc6CAgAELEDEIMBOgQIABAKOgcIIxDqAhAnOgcIABCHAhAUSgQIQRgAULy2AVibwAFg4McBaARwAngAgAF1iAHTBZIBAzEuNpgBAKABAaoBB2d3cy13aXqwAQrIAQLAAQE&sclient=gws-wiz) (GD)
  * [Backpropagation (backprop)](https://www.google.com/search?q=backpropagation&oq=backpropagation&aqs=chrome..69i57j0l6j69i61.276j0j1&sourceid=chrome&ie=UTF-8)
* prepare [an simple inference model](https://youtu.be/EyKiYVwrdjE?t=315)
  * [amortized inference](https://www.google.com/search?q=amortized+inference&sxsrf=ALeKk01ZvFMne_PiVB66PKRsCCeoS9QzBA%3A1627313206919&ei=NtT-YMTdN8_hwAOR_7mQDg&oq=amorti&gs_lcp=Cgdnd3Mtd2l6EAMYAzICCAAyAggAMgIIADICCAAyAggAMgIIADICCAAyAggAMgIIADICCAA6BwgAEEcQsAM6BAgjECc6BwgAEIcCEBQ6BQgAELEDOgoIABCHAhCxAxAUOgQIABBDOggIABCxAxCDAToHCCMQ6gIQJzoMCCMQJxCdAhBGEPoBOgoIABCxAxCDARBDSgQIQRgAUJ-wEljtyBJgweUSaANwAngAgAH4AogB2hCSAQc4LjYuMC4ymAEAoAEBqgEHZ3dzLXdperABCsgBAcABAQ&sclient=gws-wiz)
  * ```Backprop is not only used for training```
  * ```Gradient descent (GD) can be used for inference```

### 4. [Neural nets](https://youtu.be/EyKiYVwrdjE?t=722) (training & classification)

* given spiral data 
* (Q) what is the objective of classification? 
  * (A) to define ```decision boundaries```

### 5. [Space-fabric stretching](https://youtu.be/EyKiYVwrdjE?t=1080) (animation) - 차원 공간 펼치기 

* decision boundaries 확인 

### 6. [Linear Classifier 손으로 그려보기](https://youtu.be/EyKiYVwrdjE?t=1227) 

* 벡터의 차원이 낮은것 보다는 [높은 것이](https://youtu.be/EyKiYVwrdjE?t=1436) optimization problem을 풀기에 쉽다 (i.e., dimension of freedom). 
  * 벡터의 차원이 높으면 더 다양한 공간(space)을 표현할 수 있기 때문이다 
  * two-neuron 만 사용하면 2-tuple vector ⇒ 평면(planar)만 표현 됨 

### 7. [Training data](https://youtu.be/EyKiYVwrdjE?t=1566); 표기법 익히기

* design matrix: [Batch data](https://blog.naver.com/cheeryun/222084272988)표현 
* class collections  → 1-hot encoding   
* 1-hot encoding 레이블을 색깔로 표현하기 

### 8. [Fully-connected layer (FC)](https://youtu.be/EyKiYVwrdjE?t=1930)

* [matrix-multiplication](https://blog.naver.com/cheeryun/222084272988)
* 신호 연산하기; ```linear-multiplication```(inner-product) and then ```squashing```(with activation function)

### 9. [Inference](https://youtu.be/EyKiYVwrdjE?t=2346)

* 7번 8번 섹션 참고 
* 딥러닝이 성공한 이유: 차원을 ```빅뱅(Big Bang)``` (= up-sampling) 처럼 크게 키웠다가 ```짜부려트리(Squashing)```면서 양질의 특징 맵을 찾을 수 있기 때문
* (Q) Why do we go in high dimensional space? 
  * (A) Because in high dimensional space ```things are easy to move``` so that the optimization algorithms actually can work.

### 10. traning01 - [loss function](https://youtu.be/EyKiYVwrdjE?t=2531) ; Logits 

* cross-entropy → negative log-likelihood 
* 확률 값을 negative log-likelihood 로 어떻게 convexity 하게 만들까? 

### 11. training02 - [gradient descent & back-propagation](https://youtu.be/EyKiYVwrdjE?t=2840)

* Convex function 최적화 (최소 방향으로 업데이트)
* chain-rule 

***

### [실습01] - PyTorch [classification implementation](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/04-spiral_classification.ipynb)

Spiral data 포인트를 분류하는 모델 실습 

* [optimizer.zero_grad()](https://youtu.be/EyKiYVwrdjE?t=3228)를 하는 이유 
  * 이전의 기울기 값을 누적시키지 않고 매 epoch 마다 초기화 

### 12. [5-step training](https://youtu.be/EyKiYVwrdjE?t=3295)

* Feed the network with input 
* Compute the loss 
* Clear up the gradients; ```optimizer.zero_grad()```
* Backpropagation 
* Update parameters 



### 13. Linear model  vs. Two-layered network 결과 비교 

* Accuracy가 [0.504 면 좋은건가 나쁜건가](https://youtu.be/EyKiYVwrdjE?t=3414)?
  * 거의 동전 던지기 확률이라 무작위성을 지니기 때문에 나쁜 결과 



***

### [실습02] - PyTorch [regression implementation](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/05-regression.ipynb)

노이즈가 섞인 data point의 [regression model](https://youtu.be/EyKiYVwrdjE?t=3691) 찾기 



### 14. [결과 비교](https://youtu.be/EyKiYVwrdjE?t=3719)

### 15. [Regression uncertainty estimation](https://youtu.be/EyKiYVwrdjE?t=3820)

* ReLU, Sigmoid, Tanh 에 따른 결과 비교 



















