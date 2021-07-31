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



### 4. (recap) [Fully-connected(FC) layer](https://youtu.be/KvvNkE2vQVk?t=1877)

* Now we know that ```natural signals have``` this specific type of ```the three properties```; {stationarity, locality, compositionality }
* FC-layer는 1D/2D signal 의 stationarity, locality, and compositionality 특성을 있는 그대로 다룬다 



***

Convolutional neural nets 이 가지는 특성. 

* natural signal 이 가지는 stationarity, locality, and compositionality 특성을 어떻게 변환 시켜서 다룰까? 

### 5. [Locality → Sparsity](https://youtu.be/KvvNkE2vQVk?t=1990)

* how can ```locality``` induce ```sparsity```? (we can use sparsity given that our signal is local)
* 





