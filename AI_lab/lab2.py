# 1
import tensorflow as tf
import math
import numpy as np


# 2
INPUT_COUNT = 2
OUTPUT_COUNT = 2
HIDDEN_COUNT = 2
LEARNING_RATE = 0.4
MAX_STEPS = 5000


INPUT_TRAIN = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
OUTPUT_TRAIN = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# 3
inputs_placeholder = tf.placeholder("float", shape=[None, INPUT_COUNT])  # 자료형 선언만
labels_placeholder = tf.placeholder("float", shape=[None, OUTPUT_COUNT])

# 4. Define weight & biaes of hidden layer
WEIGHT_HIDDEN = tf.Variable(tf.truncated_normal([INPUT_COUNT, HIDDEN_COUNT]))
BIAS_HIDDEN = tf.Variable(tf.zeros([HIDDEN_COUNT]))

# 5
AF_HIDDEN = tf.nn.sigmoid(tf.matmul(inputs_placeholder, WEIGHT_HIDDEN) + BIAS_HIDDEN)
# matmul : 행렬곱

# 6. Define weight & biaes of output layer
WEIGHT_OUTPUT = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, OUTPUT_COUNT]))
BIAS_OUTPUT = tf.Variable(tf.zeros([OUTPUT_COUNT]))

# 7
logits = tf.matmul(AF_HIDDEN, WEIGHT_OUTPUT) + BIAS_OUTPUT

# 7a
y = tf.nn.softmax(logits)

# 8
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder)

# 8a
loss = tf.reduce_mean(cross_entropy)

# 9
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

#feed
feed_dict ={inputs_placeholder: INPUT_TRAIN, labels_placeholder: OUTPUT_TRAIN}

# 10
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    #11
    '''print every 100 steps'''
    for i in range(MAX_STEPS):
        loss_val = sess.run([train_step,loss], feed_dict)
        if i % 100 == 0:
            print('step: ', i, ',loss:',loss_val)
            for input_value in INPUT_TRAIN:
                print(input_value, sess.run(y,feed_dict={inputs_placeholder:[input_value]}))




