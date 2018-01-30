from __future__ import division, print_function, absolute_import

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 200
batch_size = 1283
display_step = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer_1
    conv1 = tf.layers.conv2d(inputs=x, filters= 32, kernel_size=[3,3], strides=1, padding='SAME', activation=tf.nn.relu)
    # Max Pooling (down-sampling)
    conv1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[2,2],strides=2, padding='same')
    # Appling dropout
    conv1 = tf.nn.dropout(conv1, dropout)

    # Convolution Layer_2
    conv2 = tf.layers.conv2d(inputs=conv1, filters= 64, kernel_size=[3,3], strides=1, padding='SAME', activation=tf.nn.relu)
    # Max Pooling (down-sampling)
    conv2 = tf.layers.max_pooling2d(inputs = conv2, pool_size=[2,2],strides=2, padding='same')
    # Appling dropout
    conv2 = tf.nn.dropout(conv2, dropout)

    # Convolution Layer_3
    conv3 = tf.layers.conv2d(inputs=conv2, filters= 128, kernel_size=[3,3], strides=1, padding='SAME', activation=tf.nn.relu)
    # Max Pooling (down-sampling)
    conv3 = tf.layers.max_pooling2d(inputs = conv3, pool_size=[2,2],strides=2, padding='same')
    # Appling dropout
    conv3 = tf.nn.dropout(conv3, dropout)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1']))

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
initializer = tf.contrib.layers.xavier_initializer()
weights = {
    'wd1': tf.Variable(initializer([4*4*128, 625])),
    # 625 inputs, 10 outputs (class prediction)
    'out': tf.Variable(initializer([625, num_classes]))
}

biases = {
    'bd1': tf.Variable(initializer([625])),
    'out': tf.Variable(initializer([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:",
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
                                      keep_prob: 1.0}))