# X 와 Y 의 상관관계를 분석하는 기초적인 선형 회귀 모델을 만들고 실행해봅니다.
import tensorflow as tf
import numpy as np


x_data = np.array([1, 2, 3])
y_data = np.array([1, 2, 3])

with tf.name_scope('aaaa'):
    W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
    W1 = W * tf.constant([[1, 0, 1]], dtype=tf.float32)
    b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
with tf.name_scope('bbbb'):
    W2 = tf.Variable(tf.truncated_normal([3, 3]))
    W3 = tf.Variable(tf.truncated_normal([3, 3]))
with tf.name_scope('cccc'):
    W4 = tf.Variable(tf.random_uniform([6, 1], -1.0, 1.0))

    b4 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# name: 나중에 텐서보드등으로 값의 변화를 추적하거나 살펴보기 쉽게 하기 위해 이름을 붙여줍니다.
X = tf.placeholder(tf.float32, [None, 1], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")
print(X)
print(Y)

# X 와 Y 의 상관 관계를 분석하기 위한 가설 수식을 작성합니다.
# y = W * x + b
# W 와 X 가 행렬이 아니므로 tf.matmul 이 아니라 기본 곱셈 기호를 사용했습니다.
L1 = tf.matmul(X, W1) + b
a = L1
b = L1
L2_1 = tf.matmul(L1, W2)
L2_2 = tf.matmul(L1, W3)
L2 = tf.concat([L2_1, L2_2], axis=1)
hypothesis = tf.matmul(L2, W4) + b4

# 손실 함수를 작성합니다.
# mean(h - Y)^2 : 예측값과 실제값의 거리를 비용(손실) 함수로 정합니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법 최적화를 수행합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 비용을 최소화 하는 것이 최종 목표
avar = tf.trainable_variables(scope='aaaa')
bvar = tf.trainable_variables(scope='bbbb')
agrad = optimizer.compute_gradients(cost, avar)
bgrad = optimizer.compute_gradients(cost, bvar)

grad = agrad + bgrad

train_op = optimizer.apply_gradients(grad)

# 세션을 생성하고 초기화합니다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: np.expand_dims(np.array([5]), axis=1)}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: np.expand_dims(np.array([2.5]), axis=1)}))
    print("\n\n")
    print(sess.run(W))
    print(sess.run(W1))
    print(sess.run(L2_1, feed_dict={X: np.expand_dims(x_data, axis=1)}))
    print(sess.run(L2_2, feed_dict={X: np.expand_dims(x_data, axis=1)}))

    # 최적화를 100번 수행합니다.
    for step in range(1000):
        # sess.run 을 통해 train_op 와 cost 그래프를 계산합니다.
        # 이 때, 가설 수식에 넣어야 할 실제값을 feed_dict 을 통해 전달합니다.
        _, cost_val = sess.run([train_op, cost], feed_dict={X: np.expand_dims(x_data, axis=1), Y: np.expand_dims(y_data, axis=1)})

        # print(step, cost_val, sess.run(W), sess.run(b))

    # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인해봅니다.
    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: np.expand_dims(np.array([5]), axis=1)}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: np.expand_dims(np.array([2.5]), axis=1)}))
    print("\n\n")
    print(sess.run(W))
    print(sess.run(W1))
    print(sess.run(L2_1, feed_dict={X: np.expand_dims(x_data, axis=1)}))
    print(sess.run(L2_2, feed_dict={X: np.expand_dims(x_data, axis=1)}))

    pass
