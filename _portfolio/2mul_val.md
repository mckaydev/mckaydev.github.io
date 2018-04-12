---
layout: post
title: Multi variable input & Load data
feature-img: "assets/img/portfolio/safe.png"
img: "assets/img/portfolio/Multi variable input & Load data.png"
date: 2013-02-15
tags: [TensorFlow]
---

```
<Multi Variable>

import tensorflow as tf

x_data = [[73., 80., 75.], [93., 88., 93.],
          [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

# x1 = tf.placeholder(tf.float32)
# x2 = tf.placeholder(tf.float32)
# x3 = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
# placeholders for a tensor that will be always fed.
# 이번 예제에선 5개의 x 데이터 셋을 입력했지만 여기선 미지수 None 으로 설정
# x 데이터 셋 안의 데이터 갯수 73 80 75 ... 3개, y 는 한개
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# w1 = tf.Variable(tf.random_normal([1]), name='weight1')
# w2 = tf.Variable(tf.random_normal([1]), name='weight2')
# w3 = tf.Variable(tf.random_normal([1]), name='weight3')
# b = tf.Variable(tf.random_normal([1]), name='bias')
# 원래는 W를 x의 갯수만큼 5개를 만들어 줘야 하지만 여기서는 [3, 1]을 통해 한문장으로
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
# b는 전과 같이 여러개의 데이터를 입력하여도 하나이기 때문에 기존과 같음
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
# 가설 또한  'x1 * w1 + x2 * w2 + x3 * w3 + b' 이 아닌 행렬 곱셈연산을 통하여 계산
hypothesis = tf.matmul(X, W) + b
# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# for step in range(2001):
#    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
#                         feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
#    if step % 10 == 0:
#        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

```


```
<Multi Variable input data>

Load-data-Prac-01.csv:
# EXAM1,EXAM2,EXAM3,FINAL
73,80,75,152
93,88,93,185
89,91,90,180
96,98,100,196
73,66,70,142
53,46,55,101

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777)  # for reproducibility

xy = np.loadtxt('Load-data-Prac-01.csv', delimiter=',', dtype=np.float32)
# [:, 0:-1] 행은 다 가져오면서 x의 데이터는 첫번째부터 마지막의 이전이니깐 -1까지
x_data = xy[:, 0:-1]
# [:, 0:-1] 행은 다 가져오면서 y의 데이터는 마지막 한 열만 이니깐 -1만 가져오기
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print("--------------------------------\n")
# x_data.shape : 배열의 모양, x_data : 배열의 내용, len(x_data) : 배열의 길이(행의 갯수)
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)
print("--------------------------------\n")

# placeholder
# (
#     placeholders for a tensor that will be always fed.

#     placeholder 자료형은 선언과 동시에 초기화 x, 일단 선언 후 그 다음 값을 전달한다.
#     따라서 반드시 실행 시 데이터가 제공되어야 한다.

#     여기서 값을 전달하는 것이 데이터를 상수값을 전달함과 같이 할당하는 것이 아니라
#     다른 텐서(Tensor)를 placeholder 에 맵핑 시키는 것이라고 보면 된다.

#     할당하기 위해 feed dictionary 를 활용, 세션을 생성시 feed_dict 의 키워드 형태로 텐서를 맵핑
#     선언 후 feed_dict 변수를 할당해도 되고 바로 값을 대입시켜도 무방.

#     dtype, : 데이터 타입을 의미하며 반드시 적어주어야 한다.
#     shape=None, : 입력 데이터의 형태를 의미한다.
#     상수 값이 될 수도 있고 다차원 배열의 정보가 들어올 수도 있다.
#     ( 디폴트 파라미터로 None 지정 )
#     name=None : 해당 placeholder 의 이름을 부여하는 것으로 적지 않아도 된다.
#     ( 디폴트 파라미터로 None 지정 )
# )
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# Variable
# (
#     텐서가 아니라 하나의 객체
#     Variable 클래스의 인스턴스가 생성되는것, 그리고 해당 인스턴스를 그래프에 추가시켜주어야 함
#
#     실제 global_variables_initializer() 를 사용(호출), 이 자체가 연산이 된다.
#     호출하기 전에 그래프의 상태는 각 노드에 값이 아직 없는 상태를 의미한다.
#     따라서 해당 함수를 사용해주어야 Variable 의 값이 할당 되는 것이고 텐서의 그래프로써의 효력이 발생
# )

# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# shape: 정수값의 1-D 텐서 또는 파이썬 배열. 반환값 텐서의 shape 입니다.
# mean: 0-D 텐서 또는 dtype 타입의 파이썬 값. 정규분포의 평균값.
# stddev: 0-D 텐서 또는 dtype 타입의 파이썬 값. 정규분포의 표준 편차.
# dtype: 반환값의 타입.
# seed: 파이썬 정수. 분포의 난수 시드값을 생성하는데에 사용됩니다. 동작 방식은 set_random_seed 를 보십시오.
# name: 연산의 명칭 (선택사항).
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
# 텐서플로우에서 행렬의 곱셈은 * 를 사용하지 않고, 텐서플로우 함수“tf.matmul”을 사용한다.
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
# 제곱 - square
# 세제곱 - cube
# N승 - to the power of N
# 제곱근 - square root
# 세제곱근 - cube root

# 1. hypothesis 방정식에서 y 좌표의 값을 빼면, 단순 거리가 나온다.
#    hypothesis - y_data 가 여기에 해당하고 hypothesis 와 y_data 모두 매트릭스. 즉, 행렬(벡터) 연산.
# 2. 단순 거리는 음수 또는 양수이기 때문에 제곱을 해서 멀리 있는 데이터에 벌점을 부여한다.
#    tf.square() - 매트릭스에 포함된 요소에 대해 각각 제곱하는 행렬 연산
# 3. 합계에 대해 평균을 계산한다.
#    tf.reduce_mean() - 합계 코드가 보이지 않아도 평균을 위해 내부적으로 합계 계산. 결과값은 실수 1개.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
# learning rate 중요하다. 이 값을 자동으로 알아낼 수는 없고, 여러 번에 걸쳐 테스트하면서 적절한 값을 찾아야 함.
# 여기서는 0.00001을 사용. 이때 0.00001이 적용되는 대상은 기울기에 해당하는 W임. 나중에 그래프가 나올 때 확인할 수 있다.
#
# gradient descent 알고리듬을 구현한 코드가 tf.train.GradientDescentOptimizer 함수.
# "경사타고 내려가기"라는 미분을 통해 최저 비용을 향해 진행하도록 만드는 핵심 함수.
# 이때 rate 를 전달했기에 W축에 대해서 매번 0.00001 만큼씩 내려가게 됨.
#
# minimize 함수는 글자 그대로 최소 비용을 찾아주는 함수라고 생각. 그러나, 정확하게는 gradient descent 알고리즘에서 gradients 를 계산,
# 변수에 적용하는 일을 동시에 하는 함수. W와 b를 적절하게 계산해서 변경하는데, 그 진행 방향이 cost 가 작아지는 쪽임.
#
# 중요한 것은 train 텐서에 연결된 것이 정말 많다는 것. optimizer 는 직접 연결되었고, optimizer 에는 cost 와 rate 가 연결되었으니까
# 이들은 한 다리 걸쳐 연결되었고, cost 에는 reduce_mean 과 square 함수를 통해 (hypothesis - y_data)의 결과가 두 다리 걸쳐
# 연결되었고, hypothesis 는 W, x_data, b와 연결되었으므로 세 다리 걸쳐 연결된 상태라는 것이다. 그래서, train 을 구한다는 것은
# 이 모든 연결된 객체들을 계산한다는 것과 같은 뜻이 된다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)
# train = optimizer.minimize(cost)

# Launch the graph in a session.
# 텐서플로우는 연산하는 과정을 그래프로 만들어서 생성하게됨. 그것이 하나의 세션을 의미
# ex) a -> x -> + -> d
#          l    l
#          b    c
# 1. Python 코드에서 텐서를 생성, 텐서를 통하여 연산을 정의.
#    이는 곧 그래프의 생성과 동시에 세션이 할당됨을 의미한다.
# 2. 생성된 세션이 연산장치(CPU, GPU) 에 의하여 연산이 할당 (Embedding) 시킨다.
# 3. 펌웨어 레벨에서 고속 연산을 수행한다. 실제로 C에서 연산을 처리하는 것이며
#    이를 통하여 파이썬의 속도한계를 극복한다.
# 4. 그리고 연산의 결과가 반환된다.
#       (Tensorflow_Architecture 참조)
#       1. Client 단 에서는 데이터의 흐름을 그래프로 정의하여 세션을 만드는 것
#       2. Distributed Master 는 Session.run() 을 통하여 그래프의 부분부분을
#          나누어 분산 처리하여 각 Worker Services 에 보낸다.
#       3. Worker Services 는 Distributed Master 단에서 받은 각 그래프의 부분조각의 작업을
#          커널(Kernel) 단에서 처리하기 위해 스케쥴링을 실시한다.
#       4. Kernel Implementations 에서는 Worker Services 에서 스케쥴된 작업 즉, 연산을 수행한다.

# 그래프를 세션에 올리기 위해 Session 객체를 만듬.
sess = tf.Session()
# Initializes global variables in the graph.
# 세션에 인자로 넘겨주고 세션의 'run()' 메소드를 호출.
# 연산이 필요로 하는 모든 입력은 세션에 의해 자동으로 실행(일반적으로는 병렬으로 실행됨)
# 자원을 시스템에 돌려주기 위해 세션을 닫아야 함. ('sess.close()')

# with tf.Session() as sess:
#   with tf.device("/gpu:1"):

# with tf.device('/cpu:0'):
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
# 이처럼 cpu 나 gpu 여러개 중에서 무엇을 사용할 것인지 정해줄 수 있음

# 변수는 그래프가 올라간 뒤 'init' 연산을 실행해서 반드시 초기화 되어야 합니다.
# 그 전에 먼저 'init' 연산을 그래프에 추가해야 합니다.
# ex)
# init_op = tf.global_variables_initializer()
# sess.run(init_op)
sess.run(tf.global_variables_initializer())

# Set up feed_dict variables inside the loop.
for step in range(5001):
    # 연산의 출력을 가져오기 위해 Session 객체에서 텐서를 run() 에 인자로 넘겨 그래프를 실행해야 함.
    # 여러 개의 텐서를 가져올 수도 있음. (ex) result = sess.run([mul, intermed])

    # 요청된 텐서들의 출력값을 만드는 데 관여하는 모든 연산은(요청된 텐서 하나 당 한 번이 x) 한 번만 실행

    # 피드(feed) 는 연산의 출력을 지정한 텐서 값으로 임시 대체합니다.
    # 임시로 사용할 피드 데이터는 run()의 인자로 넘겨줄 수 습니다.
    # 피드데이터는 run 을 호출할 때 명시적 인자로 넘겨질 때만 사용됩니다.
    # 흔한 사용법은 tf.placeholder() 를 사용해 특정 연산을 "피드" 연산으로 지정하는 것입니다.
    # + placeholder() 연산은 피드 데이터를 제공하지 않으면 오류를 일으킵니다.
    cost_val, hy_val, _ = sess.run([cost, hypothesis, optimizer], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        # step : 몇번째 러닝인지, cost_val : 현재 코스트 값, hy_val : 현재까지의 데이터로 가설 계산후 결과
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

# Ask my score
# 원하는 데이터 값을 준 뒤, 그 데이터를 가설에 넣어서 계산하라고 세션에 명령해줌.
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

# plt.plot((W[0] + W[1] + W[2]) / 3, cost_val, 'ro')
# plt.xlabel('W')
# plt.ylabel('Cost')
# plt.show()
```
