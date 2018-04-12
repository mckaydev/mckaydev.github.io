---
layout: post
title: Softmax classification
feature-img: "assets/img/portfolio/submarine.png"
img: "assets/img/portfolio/Softmax classification.png"
date: 2013-03-15
tags: [TensorFlow]
---

```
<Softmax>

import tensorflow as tf
import numpy as np

# 로지스틱과는 달리 여러개의 label 를 갖는 multinomial classification 을 손쉽게 구현
  할 수 있음
# 1과 0이 아닌 a b c중 선택지라면 로지스틱은 여러개의 biniary classification 이
  필요하지만 softmax 는 로지스틱의 행렬들을 하나의 행렬로 결합하여 사용하기 때문에
  변수 1개로 처리가 가능하다

xy = np.loadtxt('05train.txt', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:4]
y_data = xy[:, 4:]

# x의 데이터 갯수는 4개
X = tf.placeholder("float", [None, 4])
# y의 데이터 갯수는 a b c 셋중 하나이기 때문에 값을 3개 준다. a에 해당하면 1 0 0 으로
  표현한다.
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

# softmax는 binary classification을 여러 번 결합한 결과다. 예측 결과가 A, B, C 중의
  하나가 되어야 한다면, 동일한 x에 대해 A가 될 확률, B가 될 확률, C가 될 확률을 모두
  구해야 한다. x는 3번 사용되지만, A, B, C에 대해서 한 번씩 필요하니까 3번 반복된다.
  그래서, W는 예측해야 하는 숫자만큼 필요하게 된다.
# 10개 중에서 예측한다면 10개의 W가 나와야 한다.
# 여기선 x의 데이터 갯수가 4개 이기때문에 W도 4개를 예측하여야 한다.
W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)

# softmax 의 두가지 역할
# 1. 입력을 sigmoid와 마찬가지로 0과 1사이의 값으로 변환한다.
# 2. 변환된 결과에 대한 합계가 1이 되도록 만들어 준다.
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
# cost 함수는 예측한 값과 실제 값의 거리(distance, D)를 계산하는 함수로, 이 값이
  줄어드는 방향으로, 즉 entropy가 감소하는 방향으로 진행하다 보면 최저점을 만나게 된다.

# cost-entropy cost 함수 저기의 Y * tf.log(hy-)는 L x log(S)를 나타낸 것으로써
# 행렬간의 곱셈이 아닌 element-wise 곱셈이다.

# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))
# Y * tf.log(hypothesis) 결과는 행 단위로 더해야 한다.
# 그림에서 보면, 최종 cost를 계산하기 전에 행 단위로 결과를 더하고 있다.
# 이것을 가능하게 하는 옵션이 reduction_indices 매개변수다.
# 0을 전달하면 열 합계, 1을 전달하면 행 합계, 아무 것도 전달하지 않으면 전체 합계.
# 이렇게 행 단위로 더한 결과에 대해 전체 합계를 내서 평균을 구하기 위해 reduce_mean 함수가 사용됐다.
# reduction_indices has been deprecated(더이상 사용되지 않는). Better to use axis instead of.
# If the axis is not set, reduces all its dimensions.
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    # 학습한 결과를 토대로 학점을 예측하고 있는데 결과에 따라 0 1 2를 반환한다.
    # argmax 함수는 one-hot encoding 을 구현하는 텐서 함수.
      arg_max는 더이상 사용되지 않음.
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9],
                                              [1, 3, 4, 3],
                                              [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))
```
