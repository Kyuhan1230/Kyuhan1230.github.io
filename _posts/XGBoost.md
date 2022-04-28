의사결정 나무의 여러 알고리즘과 단점: **과적합에 취약**하다

### 앙상블 (Ensemble)

**앙상블 (Ensemble)**을 **통일, 조화**를 뜻하는 프랑스어로 주로 음악에서 여러 악기에 협주를 뜻하는 말

많은 수의 <font color='green'><u>작은 악기소리</u></font>가 <strong>조화</strong>를 이루어 더욱 더 웅장하고 <font color="blue"><u>아름다운 소리</u></font>를 만든다.

기계학습에서의 앙상블도 이와 비슷하다. 여러 개의 <font color='green'><u>weak learner</u></font>들이 모여 <strong>투표 (voting)</strong>를 통해 더욱 더 강력한 <font color='blue'><u>strong learner</u></font>를 구성g한다. 많은 모델이 있기 때문에, 한 모델의 예측 성능이 뒤쳐져도 어느 정도 보정된,  일반화된 모델을 구성할 수 있다.

단일 모델로는 Decision tree, SVM, Deep learning 등 모든 종류의 학습 모델이 사용될 수 있다.



#### Voting: 하드 보팅 (Hard voting)

하드 보팅은 **각 weak learner들의 예측 결과값을 바탕으로 다수결 투표**하는 방식

#### Voting: 소프트 보팅 (Soft voting)

소프트 보팅은 **weak learner들의 예측 확률값의 평균 또는 가중치 합**을 사용하는 방식
(weak learner 개별의 예측값은 중요하지 않고 예측 확률값을 단순 평균내어 확률이 더 높은 클래스를 최종 예측값으로 결정된다. 따라서, 가중치에 따라 예측값이 달라질 수 있다.)



앙상블 알고리즘은 학습 방식에 따라 크게 **배깅 (Bagging)**, **부스팅 (Boosting)**, 그리고 **스태킹 (Stacking)**으로 나눌 수 있습니다.

#### 학습방식: 배깅(Bagging)

배깅 (Bagging)은 **B**ootstrap **Agg**regat**ing**의 약자로 **부트스트랩 (Boostrap)***을 이용하는 학습 방식
부트스트랩: 주어진 데이터셋에서 random sampling 하여 새로운 데이터셋을 만들어내는 것을 의미

부트스트랩을 통해 만들어진 여러 데이터셋을 바탕으로 weak learner를 훈련시킨 뒤, 결과를 voting 한다.

#### 학습방식: 부스팅 (Boosting)

부스팅 (Boosting)은 반복적으로 모델을 업데이트하는 학습 방식.

이전 iteration의 결과에 따라 데이터셋 샘플에 대한 가중치를 부여하고 반복할 때마다 각 샘플의 중요도에 따라 다른 분류기가 만들어지게 됩니다. 최종적으로는 모든 iteration에서 생성된 모델의 결과를 voting한다.

Boosting은 다시 **Adaptive Boosting (AdaBoost)**와 **Gradient Boosting Model (GBM)** 계열로 나눌 수 있다.



#### 학습방식: 스태킹 (Stacking)

스태킹은 weak learner들의 예측 결과를 바탕으로 meta learner로 학습시켜 최종 예측값을 결정하는 것



<hr>

### Gradient Boosting for Regression

#### AdaBoost VS Gradient Boosting

AdaBoost와 Gradient Boosting의 공통점: 부스팅 앙상블 기반의 알고리즘

1. 부스팅 앙상블의 대표적인 특징은 모델 학습이 **sequential**하다. 즉, 먼저 생성된 모델의 예측값이 다음 모델 생성에 영향을 줍니다.

AdaBoost에 비교되는 Gradient Boosting의 대표적인 차이점

1. **Weak learner**: Stumps VS A leaf & Restricted trees
2. **Predicted value**: Output VS Pseudo-residual
3. **Model weight**: Different model weights (amount of say) VS Equal model weight (learning rate)



#### Gradient Boosting for Regression

Gradient Boosting은 회귀 (Regression)와 분류 (Classification) 문제에 모두 사용 모두 가능하다.

> *Create decision trees to predict residual (observed value – predicted value) of* ***__***, with limitation of maximum number of leaves

위 문장에서 빈칸에 들어가는 것에 따라 회귀와 분류로 나눌 수 있다. regression에 대한 절차는 다음과 같다.

1. Create a first leaf
2. Calculate pseudo-residuals
3. Create a next tree to predict pseudo-residuals
4. Repeat 2-3

- (Test) Scale and add up the results of each tree



<hr>

### XGBoost for Regression

XGBoost (eXtreme Gradient Boost)는 2016년 Tianqi Chen과 Carlos Guestrin 가 [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) 라는 논문으로 발표했으며, 그 전부터 Kaggle에서 놀라운 성능을 보이며 사람들에게 알려졌다.

XGBoost의 특징을 요약하면 아래와 같다.

1. Gradient Boost
2. Regularization
3. An unique regression tree
4. Approximate Greedy Algorithm
5. Parallel learning
6. Weighted Quantile Sketch
7. Sparsity-Aware Split Finiding
8. Cache-Aware Access
9. Blocks for Out-of-Core Computation

이 중 4-9번 항목은 알고리즘 효율성을 위한 최적화 방법을 나타내는 특징임.

#### XGBoost for Regression

XGBoost for Regression은 Gradient Boosting for Regression과 전체적인 순서는 동일하다.

샘플에 대한 residual을 계산하고, 이를 예측하는 decision tree를 만드는 과정을 반복한 뒤, learning rate η를 곱해 합친다. 다만, 3단계에서 달라지는 내용이 있다.

1. Create a first leaf
2. Calculate pseudo-residuals
3. Create a next tree to predict pseudo-residuals
   1. Similarity score of root node
   2. Separation based on Gain
   3. Complete decision tree with limitation of depth
   4. Prune the tree according to γγ
   5. Calculate Output value (Representative value)
4. Repeat 2-3

* (Test) Scale and add up the results of each tree



<hr>

### 변수 중요도



## 참고자료:

Guido, Sarah, Introduction to Machine Learning with Python(O'Reilly Media, 2017)

권철민, 파이썬 머신러닝 완벽가이드: 다양한 캐글 예제와 함께 기초 알고리즘부터 최신 기법까지 배우는 (위키북스, 2020)

https://tyami.github.io/machine%20learning/ensemble-6-boosting-XGBoost-regression/

[XGBoost Part 1: Regression](https://www.youtube.com/watch?v=OtD8wVaFm6E)
