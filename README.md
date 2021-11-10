## 용어정리
- ANN : 인공신경망, 뉴런 네트워크에서 영감을 받은 머신러닝 모델
- DNN : 심층신경망, 은닉층을 여러 개 쌓아 올린 인공 신경망
- 퍼셉트론 : 입력 값과 활성화 함수를 사용해 출력 값을 다음으로 넘기는 가장 작은 신경망 단위
- 활성화 함수 : 0 또는 1을 판단하는 함수
- 다층퍼셉트론 : 입력층과 출력층 사이에 은닉층을 만들어 좌표 평면을 왜곡시키는 단위
- 은닉층 : 가중치와 바이어스 값을 모아 한 번 더 시그모이드 함수를 이용해 최종 값으로 결과를 보냄
- 오차 역전파(backpropagation) : 임의의 초기 가중치를 준 뒤 결과를 계산하고 오차가 작아지는 방향으로 경사하강법을 이용해서 업데이트, 더이상 오차가 줄어들지 않을때까지 반복

#### 활성화 함수
- 기울기 소실 : 층이 늘어나면서 기울기 값이 점점 작아져 맨 처음 층까지 전달되지 않는 문제
- 하이퍼볼릭 탄젠트(tanh) : 시그모이드 함수의 범위를 -1에서 1로 확장, 1보다 작은 값이 존재하므로 기울기 소실 문제가 사라지지 않음
- 렐루(ReLU) : 0보다 작으면 0, 0보다 큰 값은 그대로, 0보다 크기만하면 미분 값이 1이므로 은닉층을 거쳐도 맨 처음 층까지 살아남음
- 소프트플러스 : 렐루의 0이 되는 순간을 완화

#### 고급 경사 하강법
- 확률적 경사 하강법(SGD) : 랜덤하게 추출한 일부 데이터를 사용하여 더 빨리 자주 업데이트, 진폭이 크고 불안정, 속도 개선
- 모멘텀(momentum) : 매번 기울기를 구하지만 오차를 수정하기 전 바로 앞 수정 값과 방향을 참고하여 같은 방향으로 일정한 비율만 수정, 정확도 개선
- 아다그라드(Adagrad) : 이동 보폭을 조절, 보폭 크기 개선
- 알엠에스프롭(RMSProp) : 아다그라드의 보폭 민감도를 보완, 보폭 크기 개선
- 아담(Adam) : 모멘텀과 알엠에스프롭 방법을 합친 방법, 정확도와 보폭 크기 개선, 현재 가장 많이 사용

## 모델링
#### loss
##### 분류
- sparse_categorical_crossentropy
- categorical_crossentropy
- binary_crossentropy
##### 회귀
- mean_squared_error

#### 클래스 불균형일 때 : fit메서드를 호출할 때 class_weight 지정
#### 성능이 만족스럽지 않을 때 
- 학습률 확인
- 다른 옵티마이저 테스트 -> 학습률 튜닝
- 층 개수, 뉴런 개수, 활성화 함수 튜닝
-> 검증 정확도가 만족스럽다면 evaluate()메서드 사용
- 테스트 세트에서 튜닝하면 일반화 오차를 매우 낙관적으로 추정
#### 콜백
- fit 메서드의 callbacks를 사용해서 훈련의 시작이나 끝에 호출할 객체 리스트 지정
```
checkpoint = keras.callbacks.ModelCheckpoint("model.h5", save_best_only=Ture)
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(x_train, y_trian, epochs=10, callbacks=[checkpoint])
```
- patience : 일정 에포크 동안 점수가 향상되지 않으면 훈련 멈춤, 최상의 가중치를 복원하므로 저장된 모델을 따로 복원할 필요 없음
#### 튜닝 : 하이퍼파라미터가 많으므로 랜덤 탐색 사용
```
from scipy.stats import reciprocal
from sklearn.model_selection import Randomized5earchCV
param_distribs = {
  "n hidden" : [0, 1, 2, 3J,
  "n neurons" np.arange( l, 100),
  "learningJate" : reciprocal(3e-4, 3e-2),
}
rnd_search_cv = Randomized5earchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[keras.callbacks.Early5topping(patience=10)J)
model = rnd_search_cv.best_estimator_.model
```
- randomizedSearchCV는 k-겹 교차 검증을 사용하기 때문에 valid를 사용하지 않고 조기종료에만 사용
- 탐색 지역이 좋다고 판명되면 더 탐색을 수행하는 여러 라이브러리 사용(ex. Hyperopt, Keras Tuner 등)
- 은닉층 뉴런 개수 : 점점 줄여서 깔때기처럼 구성, 많은 층과 뉴런을 가진 모델을 선택하고 조기 종료를 사용
- 학습률 : 최적 학습률은 최대 학습률의 절반 정도, 매우 낮은 학습률에서 시작해서 점진적으로 큰 학습률까지 반복, 손실이 다시 상승하는 지점보다 조금 아래
- 배치크기 : GPU에 맞는 가장 큰 배치 크기, 큰 배치를 써보고 불안정하거나 만족스럽지 못하면 작은 배치 
- 규제 : l1,l2,dropout,max_norm
  - l1, l2 : 연결 가중치를 제한
  - dropout : 가장 인기 있는 규제 기법, 10~50% 사이, 일부 입력을 랜덤하게 버리고 남은 입력을 보존확률로 나눔, 과대적합되면 비율을 늘림
  - max-norm : 전체 손실함수에 규제 손실항을 추가하지 않음, 연결 가중치 제한
