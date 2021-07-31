# 04.1 – [Natural signals properties and the convolution](https://youtu.be/KvvNkE2vQVk)


### 1. Topic: [convolutional neural nets](https://youtu.be/KvvNkE2vQVk?t=78)
* Exploiting ```stationarity```, ```locality```, and ```compositionality``` of natural data



### 2.  [Input layer / samples](https://youtu.be/KvvNkE2vQVk?t=157)

* what kind of data is effective for being fed to a convolution neural nets? 
* 여러가지 data type의 자료구조에 대한 설명 → 이중에서 CNN에 적합한 데이터의 구조는 무엇일까? 



### 3. [Natural signal properties](https://youtu.be/KvvNkE2vQVk?t=1118)

* How does the 1D signal work? 
  * Signals can be represented as ```vectors ```
  * [1D stationarity](https://youtu.be/KvvNkE2vQVk?t=1241) - Do similar type of patterns happen? (반복되는 비슷한 패턴 찾기)
  * [1D locality](https://youtu.be/KvvNkE2vQVk?t=1285) - there is some type of ```local structure``` in the signal (the information has a strong correlation in a local regions and then it happens again and again)
* How does the 2D signal work? 
  * [2D stationarity](https://youtu.be/KvvNkE2vQVk?t=1409) - 픽셀 사이의 틈(crevice)가 보임. 어떤 similar pattern이 happen again again 일까?  
  * [2D locality](https://youtu.be/KvvNkE2vQVk?t=1504) - the information in the signal happen to have local property as locality 
  * [2D compositionality](https://youtu.be/KvvNkE2vQVk?t=1616) - ```함께 놓여있을 때 생기는 성질```. 이미지는 픽셀들이 모여서 의미있는 형상을 만든다. (hierarchical combination of individual stimulate or signal)



### 4. (recap) [Fully-connected(FC) layer](https://youtu.be/KvvNkE2vQVk?t=1877) (a.k.a., Linear layer)

* Now we know that ```natural signals have``` this specific type of ```the three properties```; {stationarity, locality, compositionality }
* FC-layer는 1D/2D signal 의 stationarity, locality, and compositionality 특성을 있는 그대로 다룬다 
* Linear layer 는 matrix multiplication & displacement(= offset) 을 통해 [rotating & squashing](https://youtu.be/KvvNkE2vQVk?t=2256) 효과를 낸다 



***

Convolutional neural nets 이 가지는 특성. 

* natural signal 이 가지는 stationarity, locality, and compositionality 특성을 어떻게 변환 시켜서 다룰까? 

### 5. [Locality → Sparsity](https://youtu.be/KvvNkE2vQVk?t=1990) (a.k.a., sparse connection)

* how can ```locality``` induce ```sparsity```? (we can use sparsity given that our signal is local)
  * fully-connected → sparsely-connected by ```kernel ```
  * 여기서 ```receptive field(RF)``` 개념이 나온다 
    * RF simply tells about ```how many given neurons can see```from the previous layer 
      * (Q1) the receptive field for the output neuron with respect to the hidden layer (출력 뉴런에서 바로 직전의 은닉계층에 대한 RF 수는?)
      * (Q2) the receptive field for the hidden neurons with respect to the input neuron (은닉 뉴런에서 입력 뉴런에 대한 RF 수는?)
      * (Q3) What is the receptive field of the output neuron with respect to the input? (출력에서 부터 입력에 대한 RF 수는?)
    * 이러한 특성 덕분에 너 깊은 계층으로 갈 수록 볼수 있는 RF 수가 커짐 ⇒ ```global view ```
* Linear layer는 matrix multiplication 연산으로  [rotating & squashing](https://youtu.be/KvvNkE2vQVk?t=2256) 효과를 낸다면 Convolution layer 는?
  * 여전히  matrix multiplication with a lots of zeros( = [sparse matrix](https://youtu.be/KvvNkE2vQVk?t=2329) ) 으로 구현이 가능하다 
  * 아무래도 컴퓨터는  matrix multiplication 연산에 최적화가 됐으니 이중 for-문 으로 구현 하는 것 보다는 matrix multiplication 으로 구현하는 것이 더 빠르다 
* 즉,  Convolution layer 는 sparse matrix 를 통한 matrix multiplication 으로 구현되기 때문에 locality → sparsity  특성이 유도 된다:
  * matrix multiplication 덕분에 여전히 high dimension 으로 임베딩하는 rotation  특성과 
  * 특성을 다시 압축하는 squashing 특성을 가진다. 



### 6. [Stationarity → parameters sharing](https://youtu.be/KvvNkE2vQVk?t=2381)

* natural signal 의 stationarity 는 계속 비슷하게 반복되는 패턴을 의미한다 
* convolution layer 는 kernel 연산을 하기 때문에 parameter sharing 이 일어난다 
* [Parameters sharing 의 장점](https://youtu.be/KvvNkE2vQVk?t=2470):
  * faster convergence → ```같은 가중치로``` 서로 다른 신호 영역에서 손실 신호의 기울기(gradients)를 얻고 ```최적화를 진행하기``` 때문에 개별 가중치로 각각 기울기를 구하고 업데이트 할 때 보다 더 빨리 수렴시킬 수 있다. 
  * better generalization → 신호 도메인 ```전 영역에 동일한 가중치```를 사용하니 ```더욱 일반적인 패턴```을 찾을 수 있다.
  * not constrained to input size (★★★) → 입력 사이즈가 어떻든 같은 사이즈의 커널로 연산 할 수 있다 
  * kernel independence (각 커널 연산은 서로 독립적) ⇒ high parallelisation  → 각 커널에 대한 연산이 서로 독립적이니 병렬처리가 당연히 가능하다 
* [Connection sparsity 의 장점](https://youtu.be/KvvNkE2vQVk?t=2609):
  * reduced amount of computation → sparse matrix with a lot of zeros 으로 matrix multiplication 을 한다면 계산량을 대폭 줄일 수 있다 ( 0에 어떤 수를 곱하든 결과는 0이니까 연산 없이 바로 0을 return 할 수 있음)
  * (Q) Does [connection sparsity equal to try](https://youtu.be/KvvNkE2vQVk?t=2693)? → No, it just means you have zeros in your matrix multiplication 



### 7. [Kernels - 1D data](https://youtu.be/KvvNkE2vQVk?t=2713)

* 연산에 사용된 kernel 의 개수 만큼 출력 결과의 채널이 늘어남 
* kernel size 계산하는 방법 
  * 1D data uses 3D kernels-collection!
  * 이거 이해 안 되면 강의 한 번 더 보기 



### 8. [Padding - 1D data](https://youtu.be/KvvNkE2vQVk?t=3056) - 덧 대기 

* padding is necessary ```when you want to have the same dimension across the network``` even though you apply a convolution 
* padding size 계산하는 방법 
* [padding의 종류와 그 효과](https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/#summary):
  * constant padding 
  * reflection padding 
  * replication padding 
  * zero padding ([이걸 더 많이 쓰는 이유](https://youtu.be/KvvNkE2vQVk?t=3204)
* (Q) by convention, [each kernel will have odd size due to](https://youtu.be/KvvNkE2vQVk?t=3236)? 
  * 홀수여야만 커널 필터의 중심점(central point)을 얻을 수 있기 때문 

***

### 9. [ConvNet for images and tensor reshaping](https://youtu.be/KvvNkE2vQVk?t=3198)

* Standard spatial CNN diagram 
  * __Multiple layers__: 
    * convolution  - ```rotation``` 역할 
    * non-linearity (ReLU & Leaky) - ```squashing``` 역할 
    * pooling - reducing special dimensionality so that 대표값 얻음 
    * batch normalization - 수렴성을 더 좋게 만듬 
  * __Residual bypass connection__:
    * 깊은 네트워크를 학습 가능하도록 만듬 



### 10. [Pooling](https://youtu.be/KvvNkE2vQVk?t=3408) - 커널(kernel)이 보는 윈도우(window)에서 대표값 구하기 

* Lp-norm 
* maximum 
* average 
* etc. 



***

### 코드 실습 - ([ref](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/06-convnet.ipynb))

* Permuted pixels (= shuffling the pixels) - 이미지의 픽셀 순서를 마구잡이로 뒤섞어 버리면? 
  * Fully-connected network는 성능 하강이 별로 없음 
  * 반면에  ConvNet은 성능이 매우 떨어짐 
* 이를 통해 [ConvNet 은 local region에 대한 공간 정보를 학습한 다는 사실을 알 수 있음](https://youtu.be/KvvNkE2vQVk?t=3974)
* 반면 FC-network 의 경우 이미지 데이터의 픽셀 순서가 마구잡이로 바뀌었더라도 통계적인 variation 이 바뀐 것이 아니기 때문에 영향을 덜 받음 





***

### Reference 

[1] [NYU-DLSP20, website](https://atcold.github.io/pytorch-Deep-Learning/) / 



















