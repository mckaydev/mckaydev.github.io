---
layout: post
title: Logistic Regression
feature-img: "assets/img/portfolio/safe.png"
img: "assets/img/portfolio/Logistic.png"
date: 2013-03-01
tags: [TensorFlow]
---

```
<Logistic Regression>

import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 두 가지 분류를 활용할 수 있는 몇 가지 예제를 설명하고 있다.
# 스팸 메일 탐지, 페이스북 피드 표시, 신용카드 부정 사용은 두 가지 값 중의 하나를
  선택하게 된다.
# 프로그래밍에서는 이 값을 boolean 이라고 부르지만, 여기서는 쉽게 1과 0으로 구분한다.
  1은 spam, show, fraud 에 해당.
# 1과 0에 특별한 값을 할당하도록 정해진 것은 아니다. 다만 찾고자 하는 것에 1을 붙이는
  것이 일반적이다.
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
# Linear Regression 의 Wx+b라는 공식을 사용하면 W가 1/2 였을 때, x의 값이 100인 경우
  50이라는 높은 값이 나옴.
# 0과 1만을 사용해야 하는데, 범위를 벗어나는 값이 나오게 된다.
# 50보다 작으면 0, 크면 1이라고 표현하거나 1/2보다 작으면 0, 크면 1이라고 표현할 수
  있는 추가 코드가 필요함.
# 그래서 시그모이드를 사용하며 시그모이드는 Linear Regression 에서 가져온 값을 0과 1
  사이의 값으로 변환한다.
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
# 시그모이드 함수의 그래프를 보면 매끄럽지 않고 울퉁불퉁한 그래프가 나오기 때문에 그걸
  펴주지 않으면 global minimum 이 아닌 local minimum 에서 최저점으로 인식해버릴 수
  있기 때문에 새로운 코스트 함수가 필요함.
# 그래프를 피기 위해서 로그가 등장한다.
# 원래는 -log(H(x)) - (y == 1) / -log(1-H(x)) - (y == 0) 두 식이 필요한데
  둘을 합친게 아래의 코스트 식이다.
# y == 1그래프가 전체 그래프의 왼쪽을, y == 0이 오른쪽을 담당한다

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
   # Initialize TensorFlow variables
   sess.run(tf.global_variables_initializer())

   for step in range(10001):
       cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
       if step % 200 == 0:
           print(step, '\t', cost_val)

   # Accuracy report
   h, c, a = sess.run([hypothesis, predicted, accuracy],
                      feed_dict={X: x_data, Y: y_data})
   print("\nHypothesis: \n", h, "\nCorrect (Y): \n", c, "\nAccuracy: \n", a)
```
