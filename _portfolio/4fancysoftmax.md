---
layout: post
title: Fancy Softmax classification
feature-img: "assets/img/portfolio/ttt.png"
img: "assets/img/portfolio/Fancy Softmax classification.png"
date: 2013-03-16
tags: [TensorFlow]
---
```
<Fancy Softmax classification>

import tensorflow as tf
import numpy as np

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
# 모든 행과 마지막 행을 뺀 2차원 배열이 x의 값
x_data = xy[:, 0:-1]
# 모든 행과 마지막 행을 가진 배열이 y의 값
y_data = xy[:, [-1]]

# y의 종류 갯수 (동물의 개채종 숫자)
nb_classes = 7  # 0 ~ 6

# x의 속성값들. (다리의 갯수나 알을 낳는지 등등의 16개 속성값)
X = tf.placeholder(tf.float32, [None, 16])
# y의 속성값. (무슨 종이냐라는 속성값 하나밖에 존재하지 않음으로 1)
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

# one_hot 이 아니기에 최대값만 1로 설정해주고 나머지는 0으로 만드는 one_hot 으로 변환
# Y(0에서 6까지의 수)를 넣고, 데이터들이 몇개의 클래스 인지(종의 숫자)를 알려줘야 함.
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
# one_hot 으로 바꿨을 때 배열에 한차원이 추가되기 때문에 차원을 다시 하나 줄여주는 기능
# one_hot rank N을 변환시키면 rank 가 N + 1이 되어버림. 예를들어
# [[0], [3]] 을 변환시키면 [[[1000000]], [[0001000]]] 이렇게 대괄호가 추가되어 아웃풋  
  됨.
# reshape 에 one_hot 결과물을 집어 넣고 행은 '-1' 즉 알아서 하고 열은 7인 2차원 배열로  
  만들어 달라고 요구
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

# W는 x에 곱해지는 값이기 때문에 x의 열이 16이니 W의 행이 16, 거기에 무슨 종인지 대입해봐야
  하기 때문에 종의 갯수 열)
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
# Y에 더해지는 값. W와 x의 곱이 [None, nb_classes]이기 때문에 그 결과의 열에 더해질 랜덤의
  값들이 때문에 1차원 배열이며 크기는 nb_classes 인 배열로 선언하고 랜덤값 집어넣음.
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
# 기본적인 로짓 선언 초기화
logits = tf.matmul(X, W) + b
# 시그모이드가 아닌 softmax 를 사용
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# 위에 있는 기존의 Softmax 식 처럼 복잡했던 식을 문자만으로 간단하게 표현하기 위한 식.

# 코스트를 계산하기 위해 one_hot 을 적용한 Y 데이터와 로짓을 넣어줌.
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
# 위의 코스트들을 평균내고 최종적인 코스트를 완성함.
cost = tf.reduce_mean(cost_i)
# 옵티마이저에게 그래디센을 이용하여 cost 를 최소화 하라고 명령.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 우리가 예측한 값. 예측한 값(확률)을 argmax 를 이용하여 0에서 6 사이의 값으로 만들어 줌
prediction = tf.argmax(hypothesis, 1)
# 실제의 값과 비교하는 연산.
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# 맞게 예측을 한 것들을 모아서 평균을 내 accuracy 에 넣어 줌.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        # 옵티마이저에 x_y_data 를 넣어서 러닝을 시킴.
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    # flatten 이란 y의 값이 [[1],[0]]일때 [1, 0] 처럼 바꾸어 주는 기능.
    # zip 이란 pred, y_data.flatten() 가 리스트이기 때문에 각각의 엘리먼트를 p와 y에
      넘겨주기 편하게 하기 위해서 씀.
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

```
