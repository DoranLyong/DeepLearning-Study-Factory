# 04.2 – [Recurrent neural networks, vanilla and gated (LSTM)](https://youtu.be/5KSGNomPJTE)


### 1. (tip) [How to summarize papers with Notion](https://youtu.be/5KSGNomPJTE?t=15)


### 2. Why do we need to go to a [higher hidden dimension](https://youtu.be/5KSGNomPJTE?t=307)(= high dimensional space)?

* 만약 은닉에서 차원을 키우지 않고 입력과 마찬가지로 2D →2D →2D →2D →...→2D  형태로 임베딩하면 어떻게 될 까? 
* you should watch [SVD decomposition](https://youtu.be/5KSGNomPJTE?t=520) from Glibert strang and etc. 
  * [Gilbert Strang: Singular Value Decomposition](Gilbert Strang: Singular Value Decomposition)
  * [Singular Value Decomposition (SVD): Mathematical Overview](https://youtu.be/nbBvuuNVfco?t=377)
  * [주성분 분석(PCA)의 기하학적 의미, 공돌이의 수학정리노트](https://youtu.be/YEdscCNsinU?t=480)
  * [선형대수학 83강: 특잇값 분해(SVD)[쑤튜브]](https://youtu.be/gxdJYNIvOI0?t=794)



### 3. (today topic): [Recurrent neural nets](https://youtu.be/5KSGNomPJTE?t=662) - handling sequential data 

* ```Vanilla``` and ```Recurrent``` NN 
  * [Combinatorial logic](https://www.google.com/search?q=Combinatorial+logic&oq=Combinatorial+logic&aqs=chrome..69i57j0i10i30l4j0i30j0i10i30j0i10i512j0i10i30l2.213j0j1&sourceid=chrome&ie=UTF-8) 
  * [Sequential logic](https://www.google.com/search?q=sequential+logic&sxsrf=ALeKk01qZ7JbEaUkMxNsYwOcoxq1RjoEIQ%3A1629733353750&ei=6cEjYa2TLcThwAPbhAo&oq=sequen+logic&gs_lcp=Cgdnd3Mtd2l6EAMYADIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeOgcIABBHELADOgoIABBHELADEIsDSgUIPBIBMUoECEEYAVDa2AFYvd0BYI3lAWgBcAB4AIABdogB4AWSAQMyLjWYAQCgAQHIAQq4AQLAAQE&sclient=gws-wiz) (★★★) - memory 개념이 나옴 



### 4. [Vector to sequence](https://youtu.be/5KSGNomPJTE?t=972) - vec2seq

* grid 형태의 데이터는 전부 다룰 수 있음 (1D, 2D, 3D-grid so on)
* (Q) [What could be an application that uses this kind of diagram](https://youtu.be/5KSGNomPJTE?t=1075)? (vec → seq 형태로 모델링 할 수 있는 어플리케이션은?)
  * (ex) Image caption ( image → script description )



### 5. [Sequence to vector](https://youtu.be/5KSGNomPJTE?t=1383) - seq2vec 

* (ex) purchase review  → star rank point 
* [Learning to execute](https://youtu.be/5KSGNomPJTE?t=1461) - 논문 리뷰 (인간의 텍스트로 프로그래밍 하는 네트워크)
  *  you provide a sequence of text describing a python program → the answer of the program 
* (Q) [Isn't for image captioning first we need to extract its features in the case of? we have many features that we feed into the RNN.  ](https://youtu.be/5KSGNomPJTE?t=1545) - RNN에 넣기 전에 feature를 먼저 뽑아야 하는거 아님? 
  * (A) Yes, whenever you have an input which is a signal over a 2D grid, you're going to be using a CNN in order to extract information.



### 6. [Sequence to vector to sequence](https://youtu.be/5KSGNomPJTE?t=1654) - seq2vec2seq

* (ex) 자연어 번역 (영어 → context → 한글)
* [Algebraic structure of the embedding space](https://youtu.be/5KSGNomPJTE?t=1831)
* vec 부분은 마치 latent code 를 만드는 부분 같음 (autoencoder 처럼 )



### 7. [Sequence to sequence](https://youtu.be/5KSGNomPJTE?t=2126) - seq2seq

* (ex) 텍스트 자동완성? 



### 8. [Training a recurrent network](https://youtu.be/5KSGNomPJTE?t=2312) - back propagation through time (BPTT)

How do we train this recurrent neural network? ```Is the back propagation also recursion``` (*i.e.*, does the current value depend on the previous value)? <br/>

Everything is just very plain back propagation. Then, ```how do we handle the fact that there is time dependencies```?

* (sol) parameter sharing! 

* [Plain feedforward network vs. RNN  구조 비교](https://youtu.be/5KSGNomPJTE?t=2360)
  * (★★★) 적어서 한번더 정리하기 
* accumulate gradients 



### 9. [Training example](https://youtu.be/5KSGNomPJTE?t=2871) - language model 

* Batch-ification (배치화)  - 문장 시퀀스로 배치 형태로(or *chunk*) 쪼갠다 
  * Get batch
* When to stop? ??? (뭔소리인지...)



### 10. [Vanishing & exploding gradients](https://youtu.be/5KSGNomPJTE?t=3068) and gating mechanism - limitations of temporally deep nets 

* gating mechanism .... 여기도 적어서 다시 정리하자... 이해 안 돼 :s 



***

### 코드 실습 - ([ref](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/08-seq_classification.ipynb))

* ㅜㅜ RNN 넘 어렵 





***

### Reference 

[1] [NYU-DLSP20, website](https://atcold.github.io/pytorch-Deep-Learning/) / 

[2] [recurrent_neural_net_demo, github](https://github.com/llSourcell/recurrent_neural_net_demo) /  

[3] [LSTM Networks with Math, YouTube](https://www.youtube.com/watch?v=9zhrxE5PQgY&list=PL2-dafEMk2A7mu0bSksCGMJEmeddU_H4D&index=16) / 



















