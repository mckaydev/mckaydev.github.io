---
layout: post
title: MNIST
feature-img: "assets/img/portfolio/cabin.png"
img: "assets/img/portfolio/MNIST.png"
date: 2018-03-25
tags: [TensorFlow]
---

```
<MNIST>

import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
# input_data 는 mnist 튜토리얼 자료
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 0부터 9까지의 숫자들이 결과로 나와야 하기 때문에 경우의 수는 10
nb_classes = 10

# MNIST 데이터의 이미지는 28 * 28 이므로 1차원으로 보면 784개
X = tf.placeholder(tf.float32, [None, 784])
# 라벨의 갯수, 0에서 9까지의 숫자들을 서로 각각의 확률(Softmax)을 계산해야 하기 때문에
  10을 줌.
# Softmax 는 여러 경우중 확률이 제일 높은 하나의 선택지를 고르는 것 이기에 모든 경우의
  수가 포함되어야 함
Y = tf.placeholder(tf.float32, [None, nb_classes])

# x에 대한 가중치 이므로 x의 열이 784이니 W의 행이 784
W = tf.Variable(tf.random_normal([784, nb_classes]))
# 결과값이 10개가 나오니 거기에 더해줄 bias 값도 10개가 있어야 함.
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis (using softmax) - 일반적인 Softmax 함수의 가설
# 딥러닝에서는 일정기준을 만족시킬 때 활성화 되기 때문에 activation 함수라고 부른다.
# 검은색의 농도를 측정하여 계산하는게 아닌 색깔의 유무라는 단순한 구조이기에 전과 같은
  식을 사용.
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
training_epochs = 15
# 한번에 100개씩의 이미지를 처리합니다.
batch_size = 100

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    # 15번 반복하는 포문
    for epoch in range(training_epochs):
        avg_cost = 0
        # num_examples 가 반환하는 값은 55,000
        # 여기서 나누어 떨어지지 않으면 뒤쪽의 나머지 이미지들은 사용하지 않는다.
        # mnist 는 train, test, validation 3개의 데이터 셋이 있음.
        total_batch = int(mnist.train.num_examples / batch_size)

        # (55,000/100=550)번 반복하는 포문
        for i in range(total_batch):
            # next_batch 함수는 지정한 수만큼 image 와 label 을 순차적으로 반환하는 함수.
            # batch_xs.shape = (100, 784), batch_ys.shape = (100, 10)

            # def next_batch(num, data, labels):
            #     '''
            #     Return a total of `num` random samples and labels.
            #     '''
            #     idx = np.arange(0, len(data))
            #     np.random.shuffle(idx)
            #     idx = idx[:num]
            # Type 1
            #     data_shuffle = [data[i] for i in idx]
            #     labels_shuffle = [labels[i] for i in idx]
            #
            #     return np.asarray(data_shuffle), np.asarray(labels_shuffle)
            # Type 2
            #     data_shuffle = data[idx]
            #     labels_shuffle = labels[idx]
            #     labels_shuffle = np.asarray(labels_shuffle.values.reshape(len(labels_shuffle), 1))
            #
            #     return data_shuffle, labels_shuffle
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            # 100개씩 나눠서 계산하기 때문에 cost 의 값을 계속 누적시켜준다.
            # epoch 하나를 돌았을 때의 코스트를 계산하기 위하여 550으로 나눠줌.
            avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    # 학습을 다 했으니 mnist.test 데이터 셋을 가지고 테스트를 진행하는 코드.
    # 테스트 데이터에서 랜덤으로 값 하나를 뽑아옴.
    r = random.randint(0, mnist.test.num_examples - 1)
    # 그리고 [r:r + 1]을 통해서(슬라이싱) r번째의 image 와 label 하나를 가져온다.
    # argmax 는 가장 긑 값을 찾아서 1로 변화하는 one-hot encoding 알고리즘을 사용하게
      해주는 함수.
    # 이 함수를 거치면 10개의 데이터중 가장 큰 확률을 가지는 요소만 1이 되고 나머지는
      0이 된다.
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    # 784개의 1차원 배열로 되어있는 이미지를 28*28로 reshape 해준다.
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
```
